#!/bin/bash
source /opt/venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port $PORT