#!/usr/bin/env bash
# Run the suite on a GPU host reached by an SSH alias (the alias, never an address, is the parameter)
# and bring the raw records back:  calibration/suite/remote.sh <ssh-alias> [driver args...]
# The results land in calibration/<hostname-as-reported-by-the-host>/raw/.
set -euo pipefail
HOST="${1:?ssh alias}"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
REMOTE_DIR='~/work/research/AI-Infra-Co-Design-calib'
NAME=$(ssh -o BatchMode=yes "$HOST" hostname)
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE_DIR/suite $REMOTE_DIR/agentic/schema $REMOTE_DIR/raw"
rsync -az --delete --exclude build "$HERE/" "$HOST:$REMOTE_DIR/suite/"
rsync -az "$ROOT/agentic/schema/" "$HOST:$REMOTE_DIR/agentic/schema/"
ssh -o BatchMode=yes "$HOST" "export PATH=\$PATH:/usr/local/cuda/bin; cd $REMOTE_DIR && PY=\$HOME/.venv/bin/python; [ -x \$PY ] || PY=python3; \$PY suite/driver.py --out raw $*"
mkdir -p "$ROOT/calibration/$NAME/raw"
rsync -az "$HOST:$REMOTE_DIR/raw/" "$ROOT/calibration/$NAME/raw/"
echo "raw records in calibration/$NAME/raw/"
