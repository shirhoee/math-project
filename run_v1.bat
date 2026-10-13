@echo off
title 3D View Synthesis - Version 1 (FastAPI + React Parallax Client)
echo ===============================================================================
echo   3D View Synthesis Engine - VERSION 1
echo   Mathematical Foundations of 2D to 3D View Synthesis (Linear Algebra / MPI)
echo ===============================================================================
echo.
echo   [1/2] Opening browser at http://localhost:8000 ...
start http://localhost:8000
echo.
echo   [2/2] Starting FastAPI ASGI server on port 8000 ...
echo   (Press Ctrl+C to terminate the server)
echo ===============================================================================
echo.
python -m uvicorn src.api.server:app --app-dir v1 --host 127.0.0.1 --port 8000 --reload
pause
