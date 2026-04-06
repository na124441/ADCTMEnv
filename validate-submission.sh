#!/usr/bin/env bash
#
# validate-submission.sh - ADCTM Submission Prevalidator
#
# Checks that your deployed HF Space is live, the Docker image builds,
# the project submission-readiness tests pass, and openenv validate succeeds.
#
# Prerequisites:
#   - Docker
#   - curl
#   - Bash
#   - Python 3 with project dependencies installed
#   - openenv CLI (openenv-core / openenv)
#
# Run:
#   ./scripts/validate-submission.sh <ping_url> [repo_dir]
#
# Examples:
#   ./scripts/validate-submission.sh https://my-team.hf.space
#   ./scripts/validate-submission.sh https://my-team.hf.space .
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
TEST_TIMEOUT=300
OPENENV_TIMEOUT=180

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    (
      sleep "$secs"
      kill "$pid" 2>/dev/null
    ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

run_python() {
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  elif command -v python >/dev/null 2>&1; then
    python "$@"
  elif command -v py >/dev/null 2>&1; then
    py -3 "$@"
  else
    return 127
  fi
}

has_openenv_cli() {
  if command -v openenv >/dev/null 2>&1; then
    return 0
  fi

  run_python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openenv.cli') else 1)" \
    >/dev/null 2>&1
}

run_openenv_validate() {
  if command -v openenv >/dev/null 2>&1; then
    openenv validate
  else
    run_python -m openenv.cli validate
  fi
}

CLEANUP_FILES=()
cleanup() {
  rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"
}
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your Hugging Face Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  ADCTM Submission Prevalidator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_BODY=$(portable_mktemp "validate-curl-body")
CURL_ERR=$(portable_mktemp "validate-curl-err")
CLEANUP_FILES+=("$CURL_BODY" "$CURL_ERR")
HTTP_CODE=$(curl -sS -o "$CURL_BODY" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_ERR" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  if [ -s "$CURL_ERR" ]; then
    log "  curl stderr:"
    tail -20 "$CURL_ERR"
  fi
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/4: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker Desktop and ensure 'docker' is on PATH."
  stop_at "Step 2"
fi

if [ ! -f "$REPO_DIR/Dockerfile" ]; then
  fail "No Dockerfile found in repo root"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$REPO_DIR" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/4: Running submission-readiness tests${NC} ..."

if ! run_python --version >/dev/null 2>&1; then
  fail "Python 3 interpreter not found"
  hint "Install Python 3.10+ and ensure python/python3 is on PATH."
  stop_at "Step 3"
fi

if [ ! -f "$REPO_DIR/run_tests.py" ] || [ ! -f "$REPO_DIR/tests/test_submission_readiness.py" ]; then
  fail "Expected test runner files are missing"
  hint "This validator expects run_tests.py and tests/test_submission_readiness.py in the repo."
  stop_at "Step 3"
fi

TEST_OK=false
TEST_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$TEST_TIMEOUT" run_python run_tests.py tests/test_submission_readiness.py 2>&1) && TEST_OK=true

if [ "$TEST_OK" = true ]; then
  pass "Submission-readiness tests passed"
else
  fail "Submission-readiness tests failed (timeout=${TEST_TIMEOUT}s)"
  printf "%s\n" "$TEST_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/4: Running openenv validate${NC} ..."

if ! has_openenv_cli; then
  fail "openenv CLI not found"
  hint "Install it with: pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$OPENENV_TIMEOUT" run_openenv_validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed (timeout=${OPENENV_TIMEOUT}s)"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your ADCTM submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
