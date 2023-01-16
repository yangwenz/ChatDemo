FROM nvcr.io/nvidia/pytorch:22.08-py3

MAINTAINER Yang Wenzhuo <wenzhuo.yang@salesforce.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    curl \
    git ssh cmake \
    zip unzip gzip bzip2

COPY . /opt
RUN pip install --no-cache-dir -r /opt/requirements.txt

WORKDIR /opt

EXPOSE 8080
RUN chmod +x /opt/start.sh
ENTRYPOINT ["/opt/start.sh"]
