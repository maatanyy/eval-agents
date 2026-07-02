#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# judge.sh — Run LLM-as-a-judge on refusal cases
# =============================================================

# ---- Configuration (edit these) ----
JUDGE_MODEL="gemini-2.5-pro"             # judge model: e.g. gemini-2.5-pro, gpt-4o

JUDGE_PROMPT_FILE="prompts/judge_prompt.txt"

# Path to the evaluation results file produced by evaluate.sh
# Example: results/eval_gemini-2.5-flash_20240315_120000.json
RESULTS_FILE="${1:-}"

TEMPERATURE=0.0
TOP_P=1.0
TOP_K=40
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

if [[ -z "$RESULTS_FILE" ]]; then
    # Auto-detect the most recent eval results file if none provided
    RESULTS_FILE=$(ls -t results/eval_*.json 2>/dev/null | grep -v '_judged' | head -1 || true)
    if [[ -z "$RESULTS_FILE" ]]; then
        echo "Error: No results file found. Run evaluate.sh first, or pass the results file as an argument:"
        echo "  bash judge.sh results/eval_<model>_<timestamp>.json"
        exit 1
    fi
    echo "Using results file: $RESULTS_FILE"
fi

python src/judge.py \
    --results-file    "$RESULTS_FILE" \
    --judge-prompt    "$JUDGE_PROMPT_FILE" \
    --model           "$JUDGE_MODEL" \
    --temperature     "$TEMPERATURE" \
    --top-p           "$TOP_P" \
    --top-k           "$TOP_K"
