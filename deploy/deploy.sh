#!/usr/bin/env bash
# deploy.sh — Deploy Corpus RAG server to GCP VM
#
# Run from the project root on Mac:
#   ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - LanceDB data indexed at ~/corpus-rag/data/
#   - ONNX model downloaded at ~/corpus-rag/models/nomic-embed-text-v1.5/
#
# Environment variables:
#   CORPUS_API_KEY    — API key for REST endpoints (required)
#   GEMINI_API_KEY    — Gemini API key for re-ranking (optional, recommended)
#   ANTHROPIC_API_KEY — Anthropic API key for re-ranking fallback (optional)
#   SKIP_MODEL    — Set to "1" to skip model transfer (523MB)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VM_NAME="openclaw-gateway"
VM_ZONE="us-central1-c"
VM_USER="ricardorivera"
VM_PROJECT="luchoopenclaw"
REMOTE_DIR="/home/${VM_USER}/corpus-rag"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

GCP_SSH="gcloud compute ssh ${VM_NAME} --zone=${VM_ZONE} --project=${VM_PROJECT}"
GCP_SCP="gcloud compute scp --zone=${VM_ZONE} --project=${VM_PROJECT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Pre-checks
# ---------------------------------------------------------------------------

if [[ -z "${CORPUS_API_KEY:-}" ]]; then
    error "CORPUS_API_KEY not set. Export it before deploying:"
    echo "  export CORPUS_API_KEY=your-secret-key"
    exit 1
fi

if [[ ! -d "${LOCAL_DIR}/data/corpus_chunks.lance" ]]; then
    error "LanceDB data not found at ${LOCAL_DIR}/data/"
    exit 1
fi

if [[ ! -f "${LOCAL_DIR}/models/nomic-embed-text-v1.5/model.onnx" ]]; then
    error "ONNX model not found at ${LOCAL_DIR}/models/nomic-embed-text-v1.5/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Create directories on VM
# ---------------------------------------------------------------------------

info "Step 1/10: Creating directories on VM..."
$GCP_SSH --command="mkdir -p ${REMOTE_DIR}/{data,models/nomic-embed-text-v1.5}"

# ---------------------------------------------------------------------------
# Step 2: Transfer code
# ---------------------------------------------------------------------------

info "Step 2/10: Transferring code..."
$GCP_SCP \
    "${LOCAL_DIR}/indexer.py" \
    "${LOCAL_DIR}/retriever.py" \
    "${LOCAL_DIR}/mcp_server.py" \
    "${LOCAL_DIR}/server.py" \
    "${LOCAL_DIR}/config_loader.py" \
    "${LOCAL_DIR}/reindex.py" \
    "${LOCAL_DIR}/context_generator.py" \
    "${LOCAL_DIR}/summary_generator.py" \
    "${LOCAL_DIR}/rebuild_fts.py" \
    "${LOCAL_DIR}/download_onnx_model.py" \
    "${LOCAL_DIR}/analyze_corpus.py" \
    "${LOCAL_DIR}/corpus_config.yaml" \
    "${LOCAL_DIR}/pyproject.toml" \
    "${LOCAL_DIR}/README.md" \
    "${VM_NAME}:${REMOTE_DIR}/"

# ---------------------------------------------------------------------------
# Step 3: Transfer LanceDB data (~22MB)
# ---------------------------------------------------------------------------

info "Step 3/10: Transferring LanceDB data (~22MB)..."
# Remove old lance data first to avoid stale manifests/fragments from previous schema
$GCP_SSH --command="rm -rf ${REMOTE_DIR}/data/corpus_chunks.lance"
$GCP_SCP --recurse \
    "${LOCAL_DIR}/data/corpus_chunks.lance" \
    "${VM_NAME}:${REMOTE_DIR}/data/"

# ---------------------------------------------------------------------------
# Step 4: Transfer ONNX model (~523MB)
# ---------------------------------------------------------------------------

if [[ "${SKIP_MODEL:-}" == "1" ]]; then
    warn "Step 4/10: Skipping model transfer (SKIP_MODEL=1)"
else
    info "Step 4/10: Transferring ONNX model (~523MB, this may take a few minutes)..."
    $GCP_SCP \
        "${LOCAL_DIR}/models/nomic-embed-text-v1.5/model.onnx" \
        "${LOCAL_DIR}/models/nomic-embed-text-v1.5/tokenizer.json" \
        "${LOCAL_DIR}/models/nomic-embed-text-v1.5/tokenizer_config.json" \
        "${LOCAL_DIR}/models/nomic-embed-text-v1.5/special_tokens_map.json" \
        "${VM_NAME}:${REMOTE_DIR}/models/nomic-embed-text-v1.5/"
fi

# ---------------------------------------------------------------------------
# Step 5: Create venv and install dependencies
# ---------------------------------------------------------------------------

info "Step 5/10: Creating venv and installing dependencies..."
$GCP_SSH --command="
    cd ${REMOTE_DIR}
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install .
"

# ---------------------------------------------------------------------------
# Step 6: Rebuild FTS index
# ---------------------------------------------------------------------------

info "Step 6/10: Rebuilding FTS index..."
$GCP_SSH --command="
    cd ${REMOTE_DIR}
    .venv/bin/python rebuild_fts.py
"

# ---------------------------------------------------------------------------
# Step 7: Install systemd service
# ---------------------------------------------------------------------------

info "Step 7/10: Installing systemd service..."

# Create the service file content and write it on the VM
$GCP_SSH --command="
    sudo tee /etc/systemd/system/corpus-rag.service > /dev/null << 'UNIT'
[Unit]
Description=Corpus RAG Server (MCP SSE + REST API)
After=network.target

[Service]
Type=simple
User=${VM_USER}
WorkingDirectory=${REMOTE_DIR}
ExecStart=${REMOTE_DIR}/.venv/bin/python ${REMOTE_DIR}/server.py
Restart=always
RestartSec=5

Environment=CORPUS_DB_PATH=${REMOTE_DIR}/data
Environment=WCRP_ONNX_MODEL_DIR=${REMOTE_DIR}/models/nomic-embed-text-v1.5
Environment=CORPUS_API_KEY=${CORPUS_API_KEY}
Environment=CORPUS_PORT=8080
Environment=GEMINI_API_KEY=${GEMINI_API_KEY:-}
Environment=ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}

[Install]
WantedBy=multi-user.target
UNIT
    sudo systemctl daemon-reload
    sudo systemctl enable corpus-rag
"

# ---------------------------------------------------------------------------
# Step 8: Create firewall rule
# ---------------------------------------------------------------------------

info "Step 8/10: Creating GCP firewall rule for port 8080..."
if gcloud compute firewall-rules describe allow-corpus-rag --project="${VM_PROJECT}" &>/dev/null; then
    warn "Firewall rule 'allow-corpus-rag' already exists, skipping"
else
    gcloud compute firewall-rules create allow-corpus-rag \
        --project="${VM_PROJECT}" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:8080 \
        --source-ranges=0.0.0.0/0 \
        --description="Allow Corpus RAG server on port 8080"
fi

# ---------------------------------------------------------------------------
# Step 9: Start service
# ---------------------------------------------------------------------------

info "Step 9/10: Starting service..."
$GCP_SSH --command="
    sudo systemctl restart corpus-rag
    sleep 3
    sudo systemctl status corpus-rag --no-pager
"

# ---------------------------------------------------------------------------
# Step 10: Verify health
# ---------------------------------------------------------------------------

info "Step 10/10: Verifying health..."
VM_IP=$(gcloud compute instances describe "${VM_NAME}" \
    --zone="${VM_ZONE}" \
    --project="${VM_PROJECT}" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
info "VM external IP: ${VM_IP}"

# Wait for the server to be ready
sleep 2
if curl -sf "http://${VM_IP}:8080/health" | python3 -m json.tool; then
    echo ""
    info "Health check passed!"
else
    warn "Health check failed — server may still be starting"
    warn "Check logs: ${GCP_SSH} --command=\"sudo journalctl -u corpus-rag -f --no-pager -n 50\""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================"
info "Deploy complete!"
echo "============================================"
echo ""
echo "Endpoints:"
echo "  Health:      http://${VM_IP}:8080/health"
echo "  Search API:  http://${VM_IP}:8080/api/search"
echo "  MCP SSE:     http://${VM_IP}:8080/sse"
echo ""
echo "MCP config for Claude Code (.mcp.json):"
echo "  {"
echo "    \"mcpServers\": {"
echo "      \"corpus-rag-remote\": {"
echo "        \"type\": \"sse\","
echo "        \"url\": \"http://${VM_IP}:8080/sse\""
echo "      }"
echo "    }"
echo "  }"
echo ""
echo "Useful commands:"
echo "  Logs:    ${GCP_SSH} --command=\"sudo journalctl -u corpus-rag -f --no-pager -n 50\""
echo "  Memory:  ${GCP_SSH} --command=\"free -h\""
echo "  Restart: ${GCP_SSH} --command=\"sudo systemctl restart corpus-rag\""
echo "  Status:  ${GCP_SSH} --command=\"sudo systemctl status corpus-rag\""
