#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# evaluate.sh — Run evaluation against an LLM agent with tools
# =============================================================

# ---- Configuration (edit these) ----
TASK="all"                                # task to evaluate: all | scheduleConsultation | resetPassword | payInvoice | updatePersonalInformation
AGENT_MODEL="gemini-2.5-flash"            # model to evaluate: e.g. gemini-2.5-pro, gpt-4o

SYSTEM_PROMPT_FILE="prompts/system_prompt.txt"
TOOLS_FILE="tools/tools.json"
EVALUATION_SET="data/evaluation_set.json"

TEMPERATURE=0.0
TOP_P=1.0
TOP_K=40

OUTPUT_DIR="results"
# ---- End configuration ----

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from repo root .env
ENV_FILE="$(realpath "$SCRIPT_DIR/../../.env")"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "Warning: .env not found at $ENV_FILE"
fi

python src/evaluate.py \
    --task            "$TASK" \
    --model           "$AGENT_MODEL" \
    --system-prompt   "$SYSTEM_PROMPT_FILE" \
    --tools           "$TOOLS_FILE" \
    --evaluation-set  "$EVALUATION_SET" \
    --temperature     "$TEMPERATURE" \
    --top-p           "$TOP_P" \
    --top-k           "$TOP_K" \
    --output-dir      "$OUTPUT_DIR"
