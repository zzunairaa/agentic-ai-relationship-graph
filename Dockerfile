# Use a lightweight Python base image
FROM python:3.11-slim

# Set up a non-root user (Standard best practice for deployment)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Define the working directory inside the container
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files into the container
COPY --chown=user . .

# Initialize the JSON files to prevent permission errors during runtime
RUN touch graph.json extractions.json

# Expose the port (7860 is standard for HF and local testing)
EXPOSE 7860

# Command to launch the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]