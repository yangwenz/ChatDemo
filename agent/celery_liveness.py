import sys
import time
from pathlib import Path

LIVENESS_FILE = Path("/tmp/celery_worker_heartbeat")

if not LIVENESS_FILE.is_file():
    sys.exit(1)

stats = LIVENESS_FILE.stat()
time_diff = time.time() - stats.st_mtime
if time_diff > 60:
    sys.exit(1)
sys.exit(0)
