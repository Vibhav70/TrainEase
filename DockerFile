# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed dependencies
RUN pip install -r requirements.txt

# Expose the application port
EXPOSE 5000

# Command to run the app using Waitress
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "app:app"]
