#!/usr/bin/env bash
cd "$(dirname "$0")"

# Free port 8000 if occupied by stale processes
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Phát hiện cổng 8000 đang bị chiếm dụng bởi tiến trình cũ."
    echo "🔄 Đang giải phóng cổng 8000..."
    kill -9 $(lsof -ti:8000) 2>/dev/null || true
    sleep 1
fi

echo "🚀 Đang khởi chạy Backend FastAPI trên http://localhost:8000..."
python3 -m uvicorn api.main:app --reload --port 8000
