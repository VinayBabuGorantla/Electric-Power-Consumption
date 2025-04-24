# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask will run on
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]