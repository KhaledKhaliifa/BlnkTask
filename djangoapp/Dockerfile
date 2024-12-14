FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the code to the working directory
COPY . /app

# Install system and Python dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 

RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]