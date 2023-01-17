docker build -f docker/Dockerfile.backend -t chatdemo-backend:v1 .
docker build -f docker/Dockerfile.agent -t chatdemo-agent:v1 .
docker build -f docker/Dockerfile.web -t chatdemo-web:v1 .
