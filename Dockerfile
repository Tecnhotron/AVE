# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies first (as root)
RUN apt update && apt install -y ffmpeg \
    # Clean up apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'appuser' with UID 1000 and a group 'appuser' with GID 1000
# Create a home directory for the user
RUN groupadd -r appuser -g 1000 && useradd -u 1000 -r -g appuser -m -s /bin/bash -c "App User" appuser

# Set environment variables for the user's home and update PATH
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

# Set the working directory *inside the user's home*
WORKDIR $HOME/app

# Change ownership of the working directory to the new user
# Although WORKDIR creates it if it doesn't exist, explicitly ensuring ownership is good practice
RUN chown appuser:appuser $HOME/app

# Switch to the non-root user *before* copying files and installing packages
USER appuser

# Copy the requirements file (will be owned by appuser due to USER command)
COPY --chown=appuser:appuser requirements.txt .

# Install Python packages (as appuser)
# Pip installs packages into user's site-packages or uses --user implicitly
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code (owned by appuser)
COPY --chown=appuser:appuser . .

# Create the directories for uploads and final output (as appuser)
# These will automatically be owned by 'appuser' because we are running as that user
RUN mkdir -p uploads output

# Make port 7860 available
EXPOSE 7860

# Set the default command to run the application (runs as appuser)
CMD ["python", "app.py"]