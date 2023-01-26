import sys
import time
from pathlib import Path

LIVENESS_FILE = Path("/tmp/celery_worker_heartbeat")

if not LIVENESS_FILE.is_file():
    print("Celery worker liveness file does not exist.")
    sys.exit(1)

stats = LIVENESS_FILE.stat()
current_time = time.time()
time_diff = current_time - stats.st_mtime
if time_diff > 120:
    print(f"Liveness file timestamp does not matches the given constraint: "
          f"({current_time}, {stats.st_mtime}, {time_diff}).")
    sys.exit(1)
sys.exit(0)
