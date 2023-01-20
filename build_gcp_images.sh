#!/bin/bash
# gcloud builds submit . -t=$TAG --machine-type=n1-highcpu-32 --timeout=9000

gcloud builds submit . \
--config=./cloudbuild/backend.yaml \
--substitutions=_IMAGE_NAME="chatdemo-backend",_STAGE="v2" \
--timeout=9000

gcloud builds submit . \
--config=./cloudbuild/web.yaml \
--substitutions=_IMAGE_NAME="chatdemo-web",_STAGE="v5" \
--timeout=9000

gcloud builds submit . \
--config=./cloudbuild/agent.yaml \
--substitutions=_IMAGE_NAME="chatdemo-agent",_STAGE="v5" \
--machine-type=n1-highcpu-32  \
--timeout=9000
