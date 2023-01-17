FROM nvcr.io/nvidia/pytorch:22.08-py3
# FROM python:3.8-slim

MAINTAINER Yang Wenzhuo <wenzhuo.yang@salesforce.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt
RUN pip install --no-cache-dir -r /opt/requirements.txt

WORKDIR /opt

EXPOSE 8080
RUN chmod +x /opt/start.sh
