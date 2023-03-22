# Set base image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy list of frontend requirements
COPY requirements-frontend.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy files to working directory (into a newly created /app directory)
COPY app.py .

# Expose port
EXPOSE 8501

CMD ["python", "app.py"]
# Execute command when container starts
# CMD ["streamlit", "run", "app.py"]

# To build this Docker image for frontend subdir, run the following (from main directory)
# docker build -t <app-name e.g. streamlitui>:latest -f frontend/Dockerfile frontend/