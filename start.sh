#!/bin/sh
PYTHONPATH=. python ./backend/server.py &
PYTHONPATH=. streamlit run --server.port 8080 ./web/demo.py
