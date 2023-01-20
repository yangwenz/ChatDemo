#!/bin/sh
# PYTHONPATH=. python ./backend/server.py
PYTHONPATH=. gunicorn backend.server:app --bind 0.0.0.0:8081
