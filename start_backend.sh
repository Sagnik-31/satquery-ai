#!/bin/sh
set -e
cd "$(dirname "$0")/backend"
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip install -r requirements.txt
exec uvicorn main:app --host 0.0.0.0 --port 8000
