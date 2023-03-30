FROM python:3.9

# Set environment variables for PostgreSQL
ENV DB_USER=*****
ENV DB_PASSWORD=*****
ENV DB_ENDPOINT=*****
ENV DB_NAME=******

# Set environment variables for Google Cloud Storage
ENV GS_BUCKET_NAME=mlflow-model

ENV GOOGLE_PROJECT_ID=automl-380717
ENV MAIN_GOOGLE_CLOUD_PROJECT=automl-380717

# Install mlflow and psycopg2-binary
RUN pip install mlflow psycopg2 google-cloud-storage

# Expose port 5000 for the MLflow server
EXPOSE 5000

# Start the MLflow server with PostgreSQL and Google Cloud Storage
CMD mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@${DB_ENDPOINT}:5432/${DB_NAME} --default-artifact-root gs://${GS_BUCKET_NAME}
