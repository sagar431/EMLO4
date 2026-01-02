FROM dog-breed-base:latest

# Install additional dependencies for web service
RUN pip install fastapi uvicorn python-multipart

# Copy the web service code
COPY src/app.py src/app.py

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the web service
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
