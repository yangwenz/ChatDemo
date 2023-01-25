#!/bin/bash

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
--substitutions=_IMAGE_NAME="chatdemo-agent",_STAGE="v6" \
--machine-type=n1-highcpu-32  \
--timeout=9000

git clone https://github.com/triton-inference-server/fastertransformer_backend.git -b v1.4 --single-branch
cp ./docker/Dockerfile.triton fastertransformer_backend/docker/
cp ./cloudbuild/triton.yaml fastertransformer_backend/
cd fastertransformer_backend
export CONTAINER_VERSION=22.12
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}

gcloud builds submit . \
--config=./triton.yaml \
--substitutions=_IMAGE_NAME="triton_with_ft",_STAGE=${CONTAINER_VERSION} \
--machine-type=n1-highcpu-32  \
--timeout=9000
