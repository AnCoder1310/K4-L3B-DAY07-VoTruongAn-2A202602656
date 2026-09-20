#!/usr/bin/env bash
cd "$(dirname "$0")"

echo "🚀 Đang khởi chạy Backend FastAPI trên http://localhost:8001..."
python3 -m uvicorn api.main:app --reload --port 8001
