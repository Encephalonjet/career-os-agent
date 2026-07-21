# Use the Official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR/app

# Install system dependencies requires for Playwright and other utilities
RUN apt-get update && apt-get install -y
build-essential
libpq-dev
&& rm -rf/var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Playwright browser binaries (Chromium)
RUN playwright install --with-deps chromium

# Copy the rest of the application files
COPY ..


# Expose the FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]