#!/bin/bash
TAG="gcr.io/salesforce-research-internal/chatdemo"
gcloud builds submit . -t=$TAG --machine-type=n1-highcpu-32 --timeout=9000
