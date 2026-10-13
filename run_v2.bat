@echo off
title 3D View Synthesis - Version 2 (FastAPI + Three.js Mesh Viewer)
echo ===============================================================================
echo   3D View Synthesis Engine - VERSION 2
echo   FastAPI Backend + Three.js 3D Mesh Viewer
echo ===============================================================================
echo.
echo   [1/2] Opening browser at http://localhost:8001 ...
start http://localhost:8001
echo.
echo   [2/2] Starting FastAPI ASGI server on port 8001 ...
echo   (Press Ctrl+C to terminate the server)
echo ===============================================================================
echo.
python -m uvicorn src.server.main:app --app-dir v2 --host 127.0.0.1 --port 8001 --reload
pause
