# AVE - AI Video Editor

AVE is an advanced AI-powered video editing platform designed to streamline the editing process for portrait-mode, aesthetic videos (such as Instagram Reels). The application leverages Google’s Gemini generative AI, MoviePy for video/audio manipulation, and Flask for the web interface to provide a semi-automated video production process with customizable editing plans.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Workflow](#architecture--workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)

---

## Overview

AVE is built to help users generate short, high-quality, portrait-mode videos that are stylistically aligned with modern, aesthetic standards. It automatically analyzes the provided media files and leverages generative AI to create a comprehensive JSON editing plan. This plan details which segments to include, when to apply effects, and how to arrange clips, thereby greatly reducing manual work.

### How It Works

1. **File Upload and Caching:**  
   Users upload various media files (videos, audios, and images). Each file is hashed and either processed or retrieved from a cache if already available, ensuring efficient re-use of uploads.

2. **Editing Plan Generation:**  
   With the help of Google Gemini API, the system generates a detailed JSON editing plan. The plan follows a specified structure that includes clip order, start and end times, speed adjustments, optional muting, and overall color adjustments.

3. **Video Assembly:**  
   The application then processes the clips based on the generated plan. Using MoviePy and FFMPEG under the hood, clips are trimmed, adjusted (including speed and volume), concatenated, and optionally overlaid with transitions and filters.

4. **Progress and Notifications:**  
   During video processing, users can receive real-time progress updates. Additionally, the system logs all steps for debugging and auditing purposes.

---

## Features

- **AI-Assisted Editing Plan:**  
  Uses Gemini AI to generate a JSON-based editing plan that incorporates advanced parameters:
  - Clip selection based on aesthetics
  - Logical sequencing of clips
  - Optional speed adjustments and mute settings
  - Background audio specification and color adjustment hints

- **Media Upload and Caching:**  
  - Secure file uploads with configurable size limits
  - SHA256-based file hashing for deduplication and caching
  - Resilient file processing with automatic timeout handling

- **Video Processing and Assembly:**  
  - Uses MoviePy and FFMPEG for video processing
  - Supports effects like speed adjustment, volume control, and color grading
  - Designed for both low-resolution previews and full HQ processing

- **Progress Monitoring and Logging:**  
  - Detailed progress updates throughout the processing chain
  - Logging of each step (file upload, processing progress, plan generation, etc.)
  - Error handling with descriptive logging for better troubleshooting

- **User Interface & Experience:**  
  - Modern, responsive, and mobile-friendly web interface built with Flask and a custom HTML/CSS design 
  - Real-time progress indicators and potential for WebSocket integration in future releases

- **Extensibility:**  
  - Well-defined modular structure for adding new editing effects and transitions
  - Prepared for integration with asynchronous task queues (Celery/Redis) for scalability
  - Possible user authentication modules and file storage enhancements in later versions

---

## Architecture & Workflow

1. **Front-End:**  
   - A clean and modern web interface served using Flask and rendered by Jinja2 templates.
   - Supports file selection, form inputs for style description, target duration, and other processing parameters.

2. **Back-End:**  
   - **Flask Application:** Acts as the central point for handling API requests, file uploads, and processing triggers.
   - **File Caching & Upload Worker:** Uses threading to manage file uploads and caching using a SHA256 hash.
   - **Editing Plan Generation:** Interacts with the Gemini AI API to generate and validate a JSON editing plan based on the inputs.
   - **Video Assembly Engine:** Processes individual clips as per the JSON plan using MoviePy and FFMPEG for final video assembly.

3. **Logging & Cleanup:**  
   - Comprehensive logging using Python’s logging framework.
   - Cleanup functions are in place (or scheduled for future automation) for temporary files and cache entries.

---

## Installation

### Prerequisites

- Python 3.8 or above
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system’s PATH
- A valid API key for the Gemini AI service

### Clone the Repository

```bash
git clone https://github.com/yourusername/AVE.git
cd AVE
```

### Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r repos/AVE/requirements.txt
```

### Setting Up the API Key

Edit the configuration in the source code (e.g., in `main.py` or via environment variables) and replace `"YOUR_API_KEY"` with your actual Gemini API key. You may also include other configuration details such as `MODEL_NAME`, `UPLOAD_FOLDER`, and target timeouts.

---

## Usage

### Running the Application

Once you have installed the dependencies and configured your API key, you can start the server with:

```bash
python repos/AVE/main.py
```

This will start the Flask server on the configured host/port (defaults to `localhost:7860`).

### Uploading Files and Generating Videos

1. **Navigate to the Web Interface:**  
   Open your web browser and go to `http://localhost:7860`.

2. **Upload Your Media Files:**  
   Use the intuitive interface to upload videos, audio tracks, or images. Ensure the files meet the allowed formats (e.g., `mp4`, `mov`, `mp3`, `jpg`, etc.).

3. **Enter Editing Details:**  
   Fill out the form with your desired style description, target duration, and (optionally) provide a sample video to guide the AI.

4. **Submit and Monitor:**  
   Submit the form. The system will start processing, and you will receive updates about each stage:
   - File upload
   - Editing plan generation
   - Video processing and assembly

5. **Preview and Download:**  
   After processing is complete, preview your generated video directly in the browser. Download the final edited video if satisfied with the result.

---

## Configuration

- **File Uploads:**  
  Configurable settings include allowed file types and maximum file sizes.

- **Timeouts & Caching:**  
  - `MAX_WAIT_TIME` controls how long the application waits for file processing.
  - `CACHE_EXPIRY_SECONDS` determines the cache duration for uploaded files.

- **Server Settings:**  
  The Flask application configuration (like `SERVER_NAME`) is set for both local development and production deployment. Adjust as required when deploying behind proxy servers or on cloud platforms.

- **Logging:**  
  Adjust logging levels in the configuration. The current setup logs INFO and ERROR levels to give detailed runtime feedback while processing files.

---

## File Structure

```
AVE/
├── repos/
│   └── AVE/
│       ├── main.py           # Main Flask application and processing code
│       ├── requirements.txt  # Project dependencies
│       └── ...               # Other Python modules and helper functions
├── uploads/                  # Directory for user-uploaded files
├── output/                   # Directory for generated and final video output
├── templates/
│   └── index.html            # Main web interface template
└── README.md                 # This file
```

Each section of the code is modularized with clear responsibilities:
- `main.py` handles the overall video editing process, including file upload, plan generation, and video assembly.
- Templates and static files deliver a modern, responsive UI.
- Helper functions manage caching, progress updates, and error handling.

---

## Contributing

Contributions are welcome! If you want to contribute new features, improvements, or fixes:

1. **Fork the Repository:** Create your own fork and clone it locally.
2. **Create a Branch:** Use feature-specific branches (e.g., `feature-websocket-notifications`).
3. **Submit a Pull Request:** Provide detailed explanations of your changes and new features.