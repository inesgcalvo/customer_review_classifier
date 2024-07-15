#!/bin/bash

service nginx start

mlflow server \
  --host localhost \
  --port 5081
  --static-prefix /mlflow \
  --backend-store-uri
postgresql://${POSTGRES_USERNAME}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}/mlflow \
  --default-artifact-root s3://mlflow-artifact-eu-central-1/