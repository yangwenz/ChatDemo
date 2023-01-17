#!/bin/sh
PYTHONPATH=. celery -A agent.worker.app worker --loglevel=info --concurrency=1
