#!/bin/bash
# gcloud builds submit . -t=$TAG --machine-type=n1-highcpu-32 --timeout=9000

gcloud builds submit . \
--config=./cloudbuild/backend.yaml \
--substitutions=_IMAGE_NAME="chatdemo-backend",_STAGE="v1" \
--timeout=9000

gcloud builds submit . \
--config=./cloudbuild/web.yaml \
--substitutions=_IMAGE_NAME="chatdemo-web",_STAGE="v3" \
--timeout=9000

gcloud builds submit . \
--config=./cloudbuild/agent.yaml \
--substitutions=_IMAGE_NAME="chatdemo-agent",_STAGE="v2" \
--machine-type=n1-highcpu-32  \
--timeout=9000
