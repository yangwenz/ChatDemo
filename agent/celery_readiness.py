import sys
from pathlib import Path

READINESS_FILE = Path("/tmp/celery_readiness")
if not READINESS_FILE.is_file():
    print("Celery readiness file does not exist.")
    sys.exit(1)
sys.exit(0)
