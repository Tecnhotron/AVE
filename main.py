import re
import shutil
import subprocess
import tempfile
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, File # Import File type
from moviepy import *
# Correct imports for v2.0 effects
from moviepy import * # Use editor import for direct access
from moviepy.video.fx.LumContrast import LumContrast
from moviepy.video.fx.MultiplySpeed import MultiplySpeed
from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
# --- ADDED IMPORT ---
from moviepy.audio.AudioClip import CompositeAudioClip

import os, uuid
import time
import mimetypes
import json
import threading
from pathlib import Path
from flask import Flask, render_template, request, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import traceback
import logging
from typing import Dict, Any, List, Optional # Optional added
import hashlib # For file hashing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_KEY = "YOUR_API_KEY" # Replace with your actual key if needed
# Updated Model Name as requested - DO NOT CHANGE
MODEL_NAME = "gemini-2.5-pro-exp-03-25"
UPLOAD_FOLDER = 'uploads'
FINAL_OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'jpg', 'jpeg', 'png'}
MAX_UPLOAD_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB
MAX_WAIT_TIME = 300  # seconds for Gemini file processing

# --- Global State ---
progress_updates: Dict[str, Dict[str, Any]] = {}
background_tasks: Dict[str, threading.Thread] = {}
intermediate_files_registry: Dict[str, List[str]] = {} # Track intermediate files per request (still needed for potential manual cleanup)

# --- Feature 2: File Caching ---
# Cache structure: { file_hash: {'file': GeminiFileObject, 'timestamp': float} }
gemini_file_cache: Dict[str, Dict[str, Any]] = {}
cache_lock = threading.Lock()
CACHE_EXPIRY_SECONDS = 24 * 60 * 60 # Cache entries expire after 24 hours (adjust as needed)

# --- Feature 3: HQ Generation ---
# Stores details needed to re-run a request for HQ
# Structure: { request_id: {'form_data': Dict, 'file_paths': Dict} }
request_details_cache: Dict[str, Dict] = {}
# -------------------------

# Initialize API
genai.configure(api_key=API_KEY)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FINAL_OUTPUT_FOLDER'] = FINAL_OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE
# Ensure SERVER_NAME is set for url_for generation in background threads
app.config['SERVER_NAME'] = 'localhost:7860' # Or your actual server name/IP if deployed

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FINAL_OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Feature 2: File Hashing Helper ---
def get_file_hash(file_path: str) -> Optional[str]:
    """Calculates the SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(4096) # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None
# ------------------------------------

def update_progress(request_id: str, stage: str, message: str, error: str | None = None, result: Dict | None = None):
    """Update the progress status for a given request ID."""
    if request_id not in progress_updates:
        progress_updates[request_id] = {}
    progress_updates[request_id]['stage'] = stage
    progress_updates[request_id]['message'] = message
    progress_updates[request_id]['error'] = error
    progress_updates[request_id]['result'] = result
    progress_updates[request_id]['timestamp'] = time.time()
    logger.info(f"Progress Update [{request_id}] - Stage: {stage}, Message: {message}")
    if error:
        logger.error(f"Progress Error [{request_id}]: {error}")

# --- Feature 1: Modified Cleanup ---
# This function is kept for potential manual cleanup or future use,
# but it's no longer called automatically in the main flow to delete source files.
def cleanup_intermediate_files(request_id: str):
    """Remove temporary files created during processing for a specific request."""
    files_to_remove = intermediate_files_registry.pop(request_id, [])
    removed_count = 0
    failed_count = 0
    if not files_to_remove:
        logger.info(f"No intermediate files registered for cleanup for request ID: {request_id}.")
        return

    logger.info(f"Cleaning up {len(files_to_remove)} intermediate files for request ID: {request_id}...")
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed intermediate file: {file_path} [{request_id}]")
                removed_count += 1
            else:
                logger.warning(f"Intermediate file not found for removal: {file_path} [{request_id}]")
        except Exception as e:
            logger.error(f"Failed to remove intermediate file {file_path} [{request_id}]: {e}")
            failed_count += 1
    logger.info(f"Intermediate file cleanup for {request_id}: {removed_count} removed, {failed_count} failed.")
# ---------------------------------

def generate_output_path(base_folder, original_filename, suffix):
    """Generate a unique output path for a file."""
    base, ext = os.path.splitext(original_filename)
    safe_base = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in os.path.basename(base))
    timestamp = int(time.time() * 1000)
    os.makedirs(base_folder, exist_ok=True)
    new_path = os.path.join(base_folder, f"{safe_base}_{suffix}_{timestamp}{ext}")
    # Intermediate files are now tracked per request ID
    return new_path

# --- Feature 2: Modified Upload Worker (Now handles caching result) ---
def upload_thread_worker(request_id: str, file_path: str, file_hash: str, upload_results: Dict[str, Any], upload_errors: Dict[str, str]):
    """Uploads a file to Gemini API, storing results/errors. Updates cache on success."""
    global gemini_file_cache, cache_lock # Access global cache and lock

    path = Path(file_path)
    if not path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(f"Upload Error [{request_id}]: {error_msg}")
        upload_errors[file_path] = error_msg
        return

    logger.info(f"Starting upload thread for [{request_id}]: {file_path} (Hash: {file_hash[:8]}...)")
    uploaded_file = None # Initialize
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = "application/octet-stream" # Default fallback
            logger.warning(f"Could not guess mime type for {file_path}. Using {mime_type}.")

        uploaded_file = genai.upload_file(path=path, mime_type=mime_type)
        logger.info(f"Upload initiated for [{request_id}]: {file_path}. URI: {uploaded_file.uri}, Name: {uploaded_file.name}")

        logger.info(f"Waiting for processing of {uploaded_file.name} [{request_id}]...")
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > MAX_WAIT_TIME:
                error_msg = f"File processing timed out after {MAX_WAIT_TIME}s for {uploaded_file.name}"
                logger.error(f"Upload Error [{request_id}]: {error_msg}")
                upload_errors[file_path] = error_msg
                # --- Feature 1: No deletion on timeout ---
                # try:
                #     genai.delete_file(uploaded_file.name)
                #     logger.info(f"Deleted timed-out file {uploaded_file.name} [{request_id}]")
                # except Exception as e:
                #     logger.error(f"Failed to delete timed-out file {uploaded_file.name} [{request_id}]: {e}")
                # -----------------------------------------
                return # Exit thread on timeout
            time.sleep(5)
            uploaded_file = genai.get_file(name=uploaded_file.name)
            logger.info(f"File {uploaded_file.name} state [{request_id}]: {uploaded_file.state.name}")

        if uploaded_file.state.name == "ACTIVE":
            upload_results[file_path] = uploaded_file
            logger.info(f"File {uploaded_file.name} is ACTIVE [{request_id}].")
            # --- Feature 2: Update Cache ---
            with cache_lock:
                gemini_file_cache[file_hash] = {'file': uploaded_file, 'timestamp': time.time()}
                logger.info(f"Added/Updated Gemini file cache for hash {file_hash[:8]}... [{request_id}]")
            # -----------------------------
        else:
            error_msg = f"File processing failed for {uploaded_file.name}. State: {uploaded_file.state.name}"
            logger.error(f"Upload Error [{request_id}]: {error_msg}")
            upload_errors[file_path] = error_msg
            # --- Feature 1: No deletion on failure ---
            # try:
            #     genai.delete_file(uploaded_file.name)
            #     logger.info(f"Deleted failed file {uploaded_file.name} [{request_id}]")
            # except Exception as e:
            #     logger.error(f"Failed to delete failed file {uploaded_file.name} [{request_id}]: {e}")
            # ---------------------------------------

    except Exception as e:
        error_msg = f"Upload/processing failed for {file_path}: {e}"
        logger.error(f"Upload Error [{request_id}]: {error_msg}")
        traceback.print_exc()
        upload_errors[file_path] = error_msg
        # --- Feature 1: No deletion on exception ---
        # if uploaded_file and uploaded_file.name:
        #     try:
        #         genai.delete_file(uploaded_file.name)
        #         logger.info(f"Attempted deletion of file {uploaded_file.name} after exception [{request_id}]")
        #     except Exception as del_e:
        #          logger.error(f"Failed to delete file {uploaded_file.name} after exception [{request_id}]: {del_e}")
        # -----------------------------------------
# ------------------------------------

def generate_editing_plan(
    request_id: str,
    uploaded_file_references: Dict[str, File], # Use File type hint
    source_media_paths: dict, # Dict mapping type -> list of local paths
    style_description: str,
    sample_video_path: str | None,
    target_duration: float
):
    """Generates a JSON editing plan using the Gemini API."""
    update_progress(request_id, "PLANNING", "Analyzing media and generating editing plan...")
    logger.info(f"Generating Editing Plan with Gemini [{request_id}]")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Source Media Paths: {source_media_paths}")
    logger.info(f"Style Description: '{style_description}'")
    logger.info(f"Sample Video Path: {sample_video_path}")
    logger.info(f"Target Duration: {target_duration}s")

    prompt_parts = [
        "You are an AI video editor assistant specializing in creating short, aesthetic, portrait-mode videos (like Instagram Reels). Your task is to analyze the provided media files and generate a detailed JSON plan for creating a video.",
        f"The user wants a video approximately {target_duration:.1f} seconds long, suitable for portrait display (e.g., 9:16 aspect ratio).",
        f"The desired style is described as: '{style_description}'. Pay close attention to the request for *aesthetic and beautiful* shots only.",
    ]

    # Add sample video if available and successfully uploaded/cached
    if sample_video_path and sample_video_path in uploaded_file_references:
        sample_file = uploaded_file_references[sample_video_path]
        prompt_parts.extend([
            "\nHere is a sample video demonstrating the desired style:",
            sample_file, # Pass the Gemini File object directly
        ])
    elif sample_video_path:
        prompt_parts.append(f"\n(Note: A style sample video was provided '{os.path.basename(sample_video_path)}' but failed to upload/process or was not found in cache, rely on the text description.)")

    prompt_parts.append("\nAvailable source media files (use these exact paths/keys in your plan):")
    media_index = 1
    source_keys = {} # Map generated key (e.g., video_1) back to local path

    # Add videos to prompt
    for path in source_media_paths.get('videos', []):
        if path in uploaded_file_references:
            key = f"video_{media_index}"
            source_keys[key] = path
            file_obj = uploaded_file_references[path] # Get the Gemini File object
            prompt_parts.append(f"- {key}: (Video file '{os.path.basename(path)}')")
            prompt_parts.append(file_obj) # Pass the Gemini File object
            media_index += 1
        else:
            prompt_parts.append(f"- (Video file '{os.path.basename(path)}' failed upload/processing/cache, cannot use)")

    # Add audio files to prompt
    audio_index = 1
    for path in source_media_paths.get('audios', []):
        if path in uploaded_file_references:
            key = f"audio_{audio_index}"
            source_keys[key] = path
            file_obj = uploaded_file_references[path]
            prompt_parts.append(f"- {key}: (Audio file '{os.path.basename(path)}')")
            prompt_parts.append(file_obj)
            audio_index += 1
        else:
            prompt_parts.append(f"- (Audio file '{os.path.basename(path)}' failed upload/processing/cache, cannot use)")

    # Add image files to prompt
    image_index = 1
    for path in source_media_paths.get('images', []):
        if path in uploaded_file_references:
            key = f"image_{image_index}"
            source_keys[key] = path
            file_obj = uploaded_file_references[path]
            prompt_parts.append(f"- {key}: (Image file '{os.path.basename(path)}')")
            prompt_parts.append(file_obj)
            image_index += 1
        else:
            prompt_parts.append(f"- (Image file '{os.path.basename(path)}' failed upload/processing/cache, cannot use)")

    prompt_parts.append(f"""
Instruction: Create a JSON object representing the editing plan. The JSON object should strictly follow this structure:
{{
  "description": "A brief text description of the overall video edit.",
  "clips": [
    {{
      "source": "string (key of the source video file, e.g., 'video_1')",
      "start_time": "float (start time in seconds within the source video)",
      "end_time": "float (end time in seconds within the source video)",
      "order": "integer (sequence number, starting from 1)",
      "mute": "boolean (optional, default false, set to true to mute this clip's audio)",
      "speed_factor": "float (optional, default 1.0. e.g., 0.5 for slow-mo, 2.0 for fast-forward)"
    }}
    // ... more clip objects
  ],
  "background_audio": {{
    "source": "string (key of the source audio file, e.g., 'audio_1', or null if no background audio)",
    "volume_factor": "float (e.g., 0.7, or null if no audio)"
  }},
  "color_adjustments": {{ // Optional overall color adjustment
    "brightness": "float (optional, e.g., 0.1 to add brightness, -0.1 to reduce. Default 0)",
    "contrast": "float (optional, e.g., 1.1 for 10% more contrast, 0.9 for 10% less. Default 1.0)"
  }}
}}

Guidelines:
- Select ONLY short, relevant, and HIGHLY AESTHETIC/BEAUTIFUL segments from the source videos that match the style description and sample (if provided). Prioritize quality over quantity, especially for portrait display.
- The total duration of the combined clips (considering speed adjustments) should be close to the target duration ({target_duration:.1f}s).
- Order the clips logically using the 'order' field.
- Use the optional 'speed_factor' field on clips to suggest slow-motion or fast-forward where it enhances the energetic/aesthetic style. Keep factors reasonable (e.g., 0.25 to 4.0).
- Optionally suggest overall 'color_adjustments' (brightness, contrast) if it fits the mood (e.g., slightly brighter and more contrast for an energetic feel). Keep adjustments subtle.
- Respond ONLY with the JSON object, nothing else. Ensure the JSON is valid.
""")

    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            temperature=0.5 # Adjust temperature as needed
        )
    )

    raw_llm_output = None
    json_plan_text = None

    try:
        logger.info(f"Sending Prompt to Gemini for JSON Plan [{request_id}]")
        # Ensure all parts are strings or File objects
        valid_prompt_parts = []
        for part in prompt_parts:
            if isinstance(part, (str, File)):
                 valid_prompt_parts.append(part)
            else:
                 logger.warning(f"Skipping invalid type in prompt_parts: {type(part)} [{request_id}]")

        response = model.generate_content(valid_prompt_parts)
        raw_llm_output = response.text
        logger.info(f"Received Raw LLM Output (length: {len(raw_llm_output)}) [{request_id}]")

        # Attempt to directly parse as JSON first, as per response_mime_type
        try:
            plan = json.loads(raw_llm_output)
            json_plan_text = raw_llm_output # Store for potential error logging
            logger.info(f"Successfully parsed raw response as JSON [{request_id}].")
        except json.JSONDecodeError:
            logger.warning(f"Direct JSON parsing failed [{request_id}]. Trying regex extraction...")
            # Fallback to regex if direct parsing fails (e.g., if model includes ``` markers despite mime type)
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_llm_output, re.DOTALL | re.IGNORECASE)
            if match:
                json_plan_text = match.group(1).strip()
                logger.info(f"Extracted JSON block using regex [{request_id}].")
                plan = json.loads(json_plan_text)
            else:
                logger.error(f"Response is not valid JSON and does not contain ```json ... ``` markers [{request_id}].")
                raise ValueError("LLM response is not valid JSON and could not be extracted.")

        # --- Validation ---
        if not isinstance(plan, dict):
            raise ValueError("LLM response parsed, but it is not a JSON object (dictionary).")
        if 'clips' not in plan or not isinstance(plan['clips'], list):
            raise ValueError("Parsed JSON plan missing 'clips' list or it's not a list.")
        if 'background_audio' not in plan or not isinstance(plan['background_audio'], dict):
            raise ValueError("Parsed JSON plan missing 'background_audio' object or it's not an object.")
        # Optional: Validate color_adjustments structure if present
        if 'color_adjustments' in plan and not isinstance(plan['color_adjustments'], dict):
             raise ValueError("Parsed JSON plan has 'color_adjustments' but it's not an object.")

        logger.info(f"Gemini Plan Extracted and Parsed Successfully [{request_id}]")

        # --- Map source keys back to local paths ---
        for clip in plan.get('clips', []):
            key = clip.get('source')
            if key in source_keys:
                clip['source_path'] = source_keys[key] # Add local path to the clip info
            else:
                available_keys_str = ", ".join(source_keys.keys())
                raise ValueError(f"Invalid source key '{key}' found in plan['clips']. Available keys: [{available_keys_str}]")

        bg_audio = plan.get('background_audio', {})
        bg_audio_key = bg_audio.get('source')
        if bg_audio_key:
            if bg_audio_key in source_keys:
                plan['background_audio']['source_path'] = source_keys[bg_audio_key] # Add local path
            else:
                available_keys_str = ", ".join(source_keys.keys())
                raise ValueError(f"Invalid source key '{bg_audio_key}' found in plan['background_audio']. Available keys: [{available_keys_str}]")

        logger.info(f"Source keys mapped successfully [{request_id}].")
        update_progress(request_id, "PLANNING", "Editing plan generated successfully.")
        return {'status': 'success', 'plan': plan}

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse AI's plan (invalid JSON): {e}"
        logger.error(f"{error_msg} [{request_id}]")
        logger.error(f"Text attempted for JSON parsing:\n{json_plan_text if json_plan_text is not None else 'N/A'}")
        logger.error(f"Original Raw LLM Response was:\n{raw_llm_output if raw_llm_output is not None else 'Response not received'}")
        update_progress(request_id, "FAILED", "Error generating plan.", error=error_msg)
        return {'status': 'error', 'message': error_msg}

    except ValueError as e:
        error_msg = f"AI plan has invalid structure or processing failed: {e}"
        logger.error(f"{error_msg} [{request_id}]")
        logger.error(f"Original Raw LLM Response was:\n{raw_llm_output if raw_llm_output is not None else 'Response not received'}")
        update_progress(request_id, "FAILED", "Error generating plan.", error=error_msg)
        return {'status': 'error', 'message': error_msg}

    except Exception as e:
        error_msg = f"An unexpected error occurred during Gemini interaction or plan processing: {e}"
        logger.error(f"{error_msg} [{request_id}]")
        traceback.print_exc()
        logger.error(f"Original Raw LLM Response was:\n{raw_llm_output if raw_llm_output is not None else 'Response not received'}")
        update_progress(request_id, "FAILED", "Error generating plan.", error=error_msg)
        return {'status': 'error', 'message': error_msg}

# --- MODIFIED FUNCTION SIGNATURE ---
def execute_editing_plan(request_id: str, plan: dict, output_filename: str, is_preview: bool = False, mute_all_clips: bool = False) -> dict:
    """
    Executes the editing plan by writing individual processed clips to temp files
    and then concatenating them using FFMPEG for potentially faster processing.
    """
    update_progress(request_id, "STARTING", f"Starting video assembly {'(Preview Mode)' if is_preview else ''} (Global Mute: {mute_all_clips}, Method: Temp Files)...")
    logger.info(f"Executing Editing Plan [{request_id}] {'(Preview Mode)' if is_preview else ''} (Global Mute: {mute_all_clips}, Method: Temp Files)")
    logger.info(f"Output filename: {output_filename}")

    clips_data = sorted(plan.get('clips', []), key=lambda x: x.get('order', 0))
    if not clips_data:
        error_msg = 'No clips found in the editing plan.'
        update_progress(request_id, "FAILED", "Video assembly failed.", error=error_msg)
        return {'status': 'error', 'message': error_msg}

    temp_dir = None
    temp_clip_paths = []
    concat_list_path = None
    target_resolution = None # Will be set by the first valid clip, expected portrait
    target_fps = None # Will be set by the first clip, needed for consistency

    try:
        # Create a dedicated temporary directory for this request
        temp_dir = tempfile.mkdtemp(prefix=f"autovideo_{request_id}_")
        logger.info(f"Created temporary directory: {temp_dir} [{request_id}]")

        num_clips = len(clips_data)
        for i, clip_info in enumerate(clips_data):
            source_path = clip_info.get('source_path')
            start_time = clip_info.get('start_time')
            end_time = clip_info.get('end_time')
            order = clip_info.get('order')
            mute_from_plan = clip_info.get('mute', False)
            speed_factor = clip_info.get('speed_factor', 1.0)

            update_progress(request_id, "PROCESSING_CLIP", f"Processing clip {i+1}/{num_clips} (Order: {order})...")
            logger.info(f"Processing clip {order} [{request_id}]: Source='{os.path.basename(source_path)}', Start={start_time:.2f}s, End={end_time:.2f}s, PlanMute={mute_from_plan}, Speed={speed_factor:.2f}x")

            # --- Basic Validation (same as before) ---
            if not all([source_path, isinstance(start_time, (int, float)), isinstance(end_time, (int, float))]):
                logger.error(f"Missing or invalid data for clip {order}. Skipping. [{request_id}]")
                continue
            if start_time >= end_time:
                logger.warning(f"Start time ({start_time:.2f}s) >= end time ({end_time:.2f}s) for clip {order}. Skipping. [{request_id}]")
                continue
            if not os.path.exists(source_path):
                logger.error(f"Source video file not found: {source_path} for clip {order}. Skipping. [{request_id}]")
                continue
            try:
                if not isinstance(speed_factor, (int, float)) or speed_factor <= 0:
                    logger.warning(f"Invalid speed_factor ({speed_factor}) for clip {order}. Using 1.0. [{request_id}]")
                    speed_factor = 1.0
            except Exception:
                 logger.warning(f"Error processing speed_factor for clip {order}. Using 1.0. [{request_id}]")
                 speed_factor = 1.0
            # --- End Validation ---

            video = None # Define video variable outside try block for finally clause
            try:
                video = VideoFileClip(source_path)

                # --- Determine Target Resolution and FPS from first valid clip ---
                if target_resolution is None:
                    temp_res = video.size
                    temp_fps = video.fps
                    if temp_res[0] > temp_res[1]: # Landscape detected
                         logger.warning(f"First clip ({order}) appears landscape ({temp_res[0]}x{temp_res[1]}). Applying portrait rotation fix by resizing.")
                         # Set target as portrait
                         target_resolution = (temp_res[1], temp_res[0])
                    else: # Already portrait
                         target_resolution = temp_res

                    # Basic FPS check
                    if not isinstance(temp_fps, (int, float)) or temp_fps <= 0:
                         logger.warning(f"Could not determine valid FPS ({temp_fps}) from first clip {order}. Defaulting to 30. [{request_id}]")
                         target_fps = 30.0
                    else:
                         target_fps = temp_fps

                    logger.info(f"Target resolution set to {target_resolution[0]}x{target_resolution[1]} [{request_id}].")
                    logger.info(f"Target FPS set to {target_fps:.2f} [{request_id}].")
                # --- End Target Determination ---


                # --- Apply Portrait Fix / Resize to Target ---
                resized_clip = video # Start with original
                if video.size[0] > video.size[1]: # Landscape source
                    logger.warning(f"Clip {order} source is landscape ({video.size[0]}x{video.size[1]}). Resizing to target portrait {target_resolution}. [{request_id}]")
                    resized_clip = video.resized(target_resolution)
                elif video.size != target_resolution: # Portrait source but wrong size
                    logger.warning(f"Clip {order} resolution {video.size} differs from target {target_resolution}. Resizing. [{request_id}]")
                    resized_clip = video.resized(target_resolution)
                # --- End Resize ---

                vid_duration = resized_clip.duration
                if vid_duration is None:
                    logger.warning(f"Could not read duration for resized clip {order}. Skipping. [{request_id}]")
                    continue

                start_time = max(0, min(start_time, vid_duration))
                end_time = max(start_time, min(end_time, vid_duration))

                if start_time >= end_time:
                    logger.warning(f"Clamped start time ({start_time:.2f}s) >= end time ({end_time:.2f}s) for clip {order}. Skipping. [{request_id}]")
                    continue

                original_subclip_duration = end_time - start_time
                if original_subclip_duration <= 0.01: # Need a small duration
                    logger.warning(f"Calculated clip duration ({original_subclip_duration:.2f}s) is too short for clip {order}. Skipping. [{request_id}]")
                    continue

                logger.info(f"Cutting clip {order} from {start_time:.2f}s to {end_time:.2f}s [{request_id}]")
                subclip = resized_clip.subclipped(start_time, end_time)

                # --- MoviePy v2.0 Effects Application ---
                effects_to_apply = []

                # Apply speed FIRST
                if speed_factor != 1.0:
                    logger.info(f"Applying speed factor {speed_factor:.2f}x to clip {order} [{request_id}]")
                    speed_effect = MultiplySpeed(factor=speed_factor)
                    effects_to_apply.append(speed_effect)

                if effects_to_apply:
                    subclip = subclip.with_effects(effects_to_apply)
                    if subclip.duration is not None:
                        logger.info(f"Clip {order} duration after effects: {subclip.duration:.2f}s [{request_id}]")
                    else:
                        logger.warning(f"Clip {order} duration unknown after effects. [{request_id}]")
                # -----------------------------------------


                # --- Write Processed Subclip to Temporary File ---
                temp_filename = f"clip_{order:03d}_{uuid.uuid4().hex[:8]}.mp4"
                temp_output_path = os.path.join(temp_dir, temp_filename)

                # Define consistent write settings for temp files
                # Crucial for ffmpeg -c copy to work later
                temp_write_kwargs = {
                    "codec": "libx264",         # Standard codec
                    "audio_codec": "aac",       # Standard codec
                    "temp_audiofile": os.path.join(temp_dir, f"temp_audio_{order}.m4a"), # Avoid conflicts
                    "remove_temp": True,
                    "fps": target_fps,          # Ensure consistent FPS
                    "logger": "bar",             # Quieter logs for temp writes
                    # Use preview settings for potentially faster *individual* writes
                    "preset": 'ultrafast' if is_preview else 'medium',
                    "bitrate": '1000k' if is_preview else '5000k' # Adjust bitrate as needed
                }

                update_progress(request_id, "WRITING_TEMP", f"Writing temp clip {i+1}/{num_clips} (Order: {order})...")
                logger.info(f"Writing temporary clip {order} to {temp_output_path} with FPS={target_fps:.2f} [{request_id}]")

                # Ensure the subclip has audio if it's supposed to
                if not (mute_all_clips or mute_from_plan) and subclip.audio is None:
                     logger.warning(f"Clip {order} was supposed to have audio but doesn't after processing. It will be silent in the temp file. [{request_id}]")
                     # No need to explicitly add silence, ffmpeg handles missing audio streams

                subclip.write_videofile(temp_output_path, **temp_write_kwargs)
                temp_clip_paths.append(temp_output_path)
                logger.info(f"Successfully wrote temporary clip {order}. [{request_id}]")

            except Exception as e:
                logger.error(f"Error processing or writing temp clip {order} from {source_path} [{request_id}]: {e}")
                traceback.print_exc()
                # Continue to the next clip
            finally:
                # Close the MoviePy objects for this clip *immediately* to free memory
                if 'subclip' in locals() and subclip:
                    try: subclip.close()
                    except Exception as ce: logger.error(f"Error closing subclip object for clip {order} [{request_id}]: {ce}")
                if 'resized_clip' in locals() and resized_clip != video: # Avoid double close if no resize happened
                    try: resized_clip.close()
                    except Exception as ce: logger.error(f"Error closing resized_clip object for clip {order} [{request_id}]: {ce}")
                if video:
                    try: video.close()
                    except Exception as ce: logger.error(f"Error closing source video object for clip {order} [{request_id}]: {ce}")


        # --- Concatenate Temporary Clips using FFMPEG ---
        if not temp_clip_paths:
            error_msg = 'No valid temporary clips could be created.'
            update_progress(request_id, "FAILED", "Video assembly failed.", error=error_msg)
            # Cleanup is handled in the main finally block
            return {'status': 'error', 'message': error_msg}

        update_progress(request_id, "CONCATENATING", f"Concatenating {len(temp_clip_paths)} temporary clips using FFMPEG...")
        logger.info(f"Preparing to concatenate {len(temp_clip_paths)} temporary clips via FFMPEG. [{request_id}]")

        # Create the FFMPEG concat list file (using absolute paths is safer)
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for clip_path in temp_clip_paths:
                # FFMPEG requires forward slashes, even on Windows, in the concat file
                # Also escape special characters if any (though uuids shouldn't have them)
                safe_path = clip_path.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
        logger.info(f"Generated FFMPEG concat list: {concat_list_path} [{request_id}]")

        # Define the path for the *intermediate* concatenated video (before audio/color)
        concatenated_video_path = os.path.join(temp_dir, f"concatenated_{request_id}.mp4")

        # Build the FFMPEG command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output without asking
            '-f', 'concat',
            '-safe', '0',  # Allow unsafe file paths (needed for concat demuxer)
            '-i', concat_list_path,
            '-c', 'copy',  # CRITICAL: Copy streams without re-encoding (FAST!)
            '-fflags', '+igndts', # Ignore DTS issues that can arise from concat
            '-map_metadata', '-1', # Avoid metadata issues from source clips
            '-movflags', '+faststart', # Good practice for web video
            concatenated_video_path
        ]

        logger.info(f"Executing FFMPEG command: {' '.join(ffmpeg_cmd)} [{request_id}]")
        try:
            # Use stderr=subprocess.PIPE to capture FFMPEG output for logging/debugging
            process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info(f"FFMPEG concatenation successful. Output:\n{process.stdout}\n{process.stderr} [{request_id}]")
        except subprocess.CalledProcessError as e:
            error_msg = f"FFMPEG concatenation failed with exit code {e.returncode}."
            logger.error(error_msg + f" [{request_id}]")
            logger.error(f"FFMPEG stderr:\n{e.stderr}")
            logger.error(f"FFMPEG stdout:\n{e.stdout}")
            update_progress(request_id, "FAILED", "FFMPEG concatenation failed.", error=error_msg + f" Details: {e.stderr[:200]}...") # Limit error length
            # Cleanup handled in finally
            return {'status': 'error', 'message': error_msg, 'details': e.stderr}
        except FileNotFoundError:
            error_msg = "FFMPEG command not found. Ensure FFMPEG is installed and in the system's PATH."
            logger.error(error_msg + f" [{request_id}]")
            update_progress(request_id, "FAILED", "FFMPEG not found.", error=error_msg)
            return {'status': 'error', 'message': error_msg}


        # --- Post-Processing: Background Audio & Color Adjustments ---
        final_processed_path = concatenated_video_path # Start with the ffmpeg output
        needs_final_write = False # Flag if we need another MoviePy write step

        bg_audio_info = plan.get('background_audio')
        color_adjustments = plan.get('color_adjustments')

        if bg_audio_info or (color_adjustments and isinstance(color_adjustments, dict)):
            needs_final_write = True
            update_progress(request_id, "POST_PROCESSING", "Applying background audio and/or color adjustments...")
            logger.info(f"Loading concatenated video for post-processing (BG Audio/Color). [{request_id}]")

            post_process_clip = None
            bg_audio_clip_to_close = None
            try:
                post_process_clip = VideoFileClip(concatenated_video_path)

                # --- Background Audio ---
                bg_audio_path = bg_audio_info.get('source_path') if bg_audio_info else None
                final_audio = post_process_clip.audio # Get audio from the concatenated clip

                if bg_audio_path and os.path.exists(bg_audio_path):
                    volume = bg_audio_info.get('volume_factor', 0.7)
                    if not isinstance(volume, (int, float)) or not (0 <= volume <= 1.5):
                        logger.warning(f"Invalid background audio volume factor ({volume}). Using default 0.7. [{request_id}]")
                        volume = 0.7

                    logger.info(f"Adding background audio: '{os.path.basename(bg_audio_path)}' with volume {volume:.2f} [{request_id}]")
                    try:
                        bg_audio_clip = AudioFileClip(bg_audio_path)
                        bg_audio_clip_to_close = bg_audio_clip # Ensure cleanup

                        target_vid_duration = post_process_clip.duration
                        if target_vid_duration is not None:
                            if bg_audio_clip.duration > target_vid_duration:
                                bg_audio_clip = bg_audio_clip.subclipped(0, target_vid_duration)
                            # Ensure bg audio matches video duration exactly if possible
                            bg_audio_clip = bg_audio_clip.set_duration(target_vid_duration)

                        processed_bg_audio = bg_audio_clip.fx(MultiplyVolume, volume)

                        if final_audio: # If the concatenated video has audio
                            logger.info(f"Compositing background audio with existing clip audio. [{request_id}]")
                            # Ensure original audio matches duration
                            if final_audio.duration != target_vid_duration:
                                logger.warning(f"Original concatenated audio duration ({final_audio.duration:.2f}s) doesn't match video duration ({target_vid_duration:.2f}s). Adjusting. [{request_id}]")
                                final_audio = final_audio.set_duration(target_vid_duration)
                            final_audio = CompositeAudioClip([final_audio, processed_bg_audio])
                        else: # If concatenated video was silent
                            logger.info(f"Setting background audio (concatenated video was silent). [{request_id}]")
                            final_audio = processed_bg_audio

                        if final_audio and target_vid_duration:
                            final_audio = final_audio.set_duration(target_vid_duration) # Final duration check

                        post_process_clip = post_process_clip.set_audio(final_audio)

                    except Exception as audio_e:
                        logger.warning(f"Failed to add background audio during post-processing [{request_id}]: {audio_e}. Proceeding without it.")
                        traceback.print_exc()
                        if post_process_clip.audio is None:
                            logger.warning(f"Final video might be silent after failed BG audio add. [{request_id}]")

                elif bg_audio_path:
                    logger.warning(f"Background audio file specified ('{os.path.basename(bg_audio_path)}') not found. Skipping BG audio. [{request_id}]")
                elif post_process_clip.audio is None:
                     logger.warning(f"No background audio specified and concatenated video is silent. Final video will be silent. [{request_id}]")


                # --- Apply Overall Color Adjustments ---
                if color_adjustments and isinstance(color_adjustments, dict):
                    raw_brightness = color_adjustments.get('brightness', 0)
                    raw_contrast = color_adjustments.get('contrast', 1.0)
                    lum_param = 0
                    contrast_param = 0.0
                    apply_color_fx = False
                    try:
                        if isinstance(raw_brightness, (int, float)) and -1.0 <= raw_brightness <= 1.0 and raw_brightness != 0:
                            lum_param = int(raw_brightness * 255)
                            apply_color_fx = True
                        if isinstance(raw_contrast, (int, float)) and 0.1 <= raw_contrast <= 3.0 and raw_contrast != 1.0:
                            contrast_param = raw_contrast - 1.0
                            apply_color_fx = True

                        if apply_color_fx:
                            logger.info(f"Applying overall color adjustments: Brightness={raw_brightness:.2f}, Contrast={raw_contrast:.2f} [{request_id}]")
                            # Moviepy 2.x syntax
                            post_process_clip = post_process_clip.fx(LumContrast, lum=lum_param, contrast=contrast_param)
                            # Moviepy 1.x was: post_process_clip = post_process_clip.fx(vfx.lum_contrast, lum=lum_param, contrast=contrast_param)

                    except Exception as e:
                        logger.error(f"Error applying color adjustments during post-processing: {e} [{request_id}]")
                        traceback.print_exc()

                # --- Define Final Output Path and Write ---
                final_output_path = os.path.join(FINAL_OUTPUT_FOLDER, secure_filename(output_filename))
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                final_processed_path = final_output_path # Update the path to the *actual* final file

                update_progress(request_id, "WRITING_FINAL", f"Writing final video with post-processing to {os.path.basename(final_output_path)} {'(Preview)' if is_preview else ''}...")
                logger.info(f"Writing final video (post-processed) to: {final_output_path} [{request_id}] {'(Preview)' if is_preview else ''}")

                final_write_kwargs = {
                    "codec": "libx264",
                    "audio_codec": "aac",
                    "threads": 16,
                    "logger": 'bar', # Show progress bar for final write
                    "preset": 'ultrafast' if is_preview else 'medium',
                    "bitrate": '500k' if is_preview else '50000k' # Use preview/final settings here
                }
                post_process_clip.write_videofile(final_output_path, **final_write_kwargs)
                logger.info(f"Successfully wrote final post-processed video. [{request_id}]")

            except Exception as post_e:
                 error_msg = f"Failed during post-processing (audio/color) or final write: {post_e}"
                 logger.error(f"Error during post-processing or final write [{request_id}]: {post_e}")
                 traceback.print_exc()
                 update_progress(request_id, "FAILED", "Post-processing/Final Write failed.", error=error_msg)
                 # Cleanup handled in finally
                 return {'status': 'error', 'message': error_msg}
            finally:
                 # Clean up post-processing clips
                 if post_process_clip:
                     try: post_process_clip.close()
                     except Exception as ce: logger.error(f"Error closing post_process_clip [{request_id}]: {ce}")
                 if bg_audio_clip_to_close:
                     try: bg_audio_clip_to_close.close()
                     except Exception as ce: logger.error(f"Error closing bg_audio_clip_to_close [{request_id}]: {ce}")

        else:
            # No post-processing needed, the FFMPEG output is the final output.
            # Move/Rename it to the final destination.
            final_output_path = os.path.join(FINAL_OUTPUT_FOLDER, secure_filename(output_filename))
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            logger.info(f"No post-processing needed. Moving concatenated file to final destination: {final_output_path} [{request_id}]")
            try:
                shutil.move(concatenated_video_path, final_output_path)
                final_processed_path = final_output_path # Update to the final path
                logger.info(f"Successfully moved concatenated video to final path. [{request_id}]")
            except Exception as move_e:
                error_msg = f"Failed to move concatenated video to final destination: {move_e}"
                logger.error(error_msg + f" [{request_id}]")
                update_progress(request_id, "FAILED", "Failed to finalize video.", error=error_msg)
                # Cleanup handled in finally
                return {'status': 'error', 'message': error_msg}


        logger.info(f"Plan Execution Successful [{request_id}]")
        update_progress(request_id, "COMPLETED", f"Video assembly complete: {os.path.basename(final_processed_path)}")
        return {'status': 'success', 'output_path': final_processed_path}

    except Exception as e:
        # Catch-all for unexpected errors during setup or flow control
        error_msg = f"An unexpected error occurred during video processing: {e}"
        logger.error(f"Unexpected error in execute_editing_plan [{request_id}]: {e}")
        logger.error(f"Error Type: {type(e).__name__}")
        traceback.print_exc()
        update_progress(request_id, "FAILED", "Unexpected processing error.", error=error_msg)
        return {'status': 'error', 'message': error_msg}

    finally:
        # --- Cleanup Temporary Files and Directory ---
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir} [{request_id}]")
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Successfully removed temporary directory. [{request_id}]")
            except Exception as cleanup_e:
                logger.error(f"Error removing temporary directory {temp_dir} [{request_id}]: {cleanup_e}")
        # Note: MoviePy clip objects should have been closed within the loop or post-processing block
        
# --- Feature 2: Cache Cleanup Function ---
def cleanup_expired_cache():
    """Removes expired entries from the Gemini file cache."""
    global gemini_file_cache, cache_lock
    now = time.time()
    expired_hashes = []
    with cache_lock:
        for file_hash, data in gemini_file_cache.items():
            if now - data.get('timestamp', 0) > CACHE_EXPIRY_SECONDS:
                expired_hashes.append(file_hash)

        if expired_hashes:
            logger.info(f"Cleaning up {len(expired_hashes)} expired Gemini cache entries...")
            for file_hash in expired_hashes:
                # --- Feature 1: No Deletion from Gemini API ---
                # cached_file = gemini_file_cache[file_hash].get('file')
                # if cached_file and hasattr(cached_file, 'name') and cached_file.name:
                #     try:
                #         genai.delete_file(cached_file.name)
                #         logger.info(f"Deleted expired Gemini file from cache: {cached_file.name} (Hash: {file_hash[:8]}...)")
                #     except Exception as del_e:
                #         logger.error(f"Failed to delete expired Gemini file {cached_file.name} from cache: {del_e}")
                # else:
                #      logger.warning(f"Could not delete expired Gemini file for hash {file_hash[:8]}... (File object missing or invalid)")
                # ---------------------------------------------
                del gemini_file_cache[file_hash]
                logger.info(f"Removed expired cache entry for hash: {file_hash[:8]}...")
            logger.info("Expired Gemini cache cleanup finished.")
# ------------------------------------

# --- Feature 3: Request Details Cache Cleanup ---
def cleanup_expired_request_details():
    """Removes old request details from the cache."""
    global request_details_cache
    expiry_time = time.time() - (48 * 60 * 60)
    ids_to_remove = [
        req_id for req_id, details in request_details_cache.items()
        if details.get('timestamp', 0) < expiry_time
    ]
    if ids_to_remove:
        logger.info(f"Cleaning up {len(ids_to_remove)} expired request details entries...")
        for req_id in ids_to_remove:
            if req_id in request_details_cache:
                del request_details_cache[req_id]
                logger.info(f"Removed expired request details for ID: {req_id}")
        logger.info("Expired request details cleanup finished.")
# ------------------------------------------

def process_video_request(request_id: str, form_data: Dict, file_paths: Dict, app: Flask):
    """
    The main logic for processing a video request, run in a background thread.
    Handles file caching and avoids deleting files.
    """
    global intermediate_files_registry, progress_updates, gemini_file_cache, cache_lock

    uploaded_file_references: Dict[str, File] = {}
    upload_errors: Dict[str, str] = {}
    upload_threads = []

    try:
        # Extract data needed for processing
        style_description = form_data.get('style_desc', '')
        target_duration = form_data.get('duration')
        output_filename = form_data.get('output')
        is_preview = form_data.get('is_preview', False)
        # --- ADDED: Get the global mute flag ---
        mute_all_clips_flag = form_data.get('mute_audio', False)
        # ---------------------------------------
        style_sample_path = file_paths.get('style_sample')
        source_media_paths = file_paths.get('sources', {})

        # --- 1. Identify files for Gemini API & Check Cache ---
        files_to_upload_api = []
        files_requiring_api = []
        if style_sample_path:
            files_requiring_api.append(style_sample_path)
        files_requiring_api.extend(source_media_paths.get('videos', []))
        files_requiring_api.extend(source_media_paths.get('audios', []))
        files_requiring_api.extend(source_media_paths.get('images', []))

        if not source_media_paths.get('videos'):
             raise ValueError("No source videos provided for processing.")

        logger.info(f"Checking cache for {len(files_requiring_api)} files potentially needing API processing [{request_id}]...")
        update_progress(request_id, "PREPARING", "Checking file cache...")

        # --- Feature 2: Cache Check ---
        cleanup_expired_cache()
        with cache_lock:
            for file_path in files_requiring_api:
                file_hash = get_file_hash(file_path)
                if not file_hash:
                    logger.warning(f"Could not calculate hash for {file_path}. Will attempt upload. [{request_id}]")
                    files_to_upload_api.append((file_path, None))
                    continue

                cached_data = gemini_file_cache.get(file_hash)
                if cached_data:
                    cached_file = cached_data.get('file')
                    try:
                        retrieved_file = genai.get_file(name=cached_file.name)
                        if retrieved_file.state.name == "ACTIVE":
                            logger.info(f"Cache HIT: Using cached Gemini file '{cached_file.name}' for {os.path.basename(file_path)} (Hash: {file_hash[:8]}...) [{request_id}]")
                            uploaded_file_references[file_path] = retrieved_file
                            gemini_file_cache[file_hash]['timestamp'] = time.time()
                        else:
                            logger.warning(f"Cache INVALID: Cached Gemini file '{cached_file.name}' for {os.path.basename(file_path)} is no longer ACTIVE (State: {retrieved_file.state.name}). Will re-upload. [{request_id}]")
                            files_to_upload_api.append((file_path, file_hash))
                            del gemini_file_cache[file_hash]
                    except Exception as get_err:
                        logger.warning(f"Cache CHECK FAILED: Error verifying cached Gemini file '{cached_file.name}' for {os.path.basename(file_path)}: {get_err}. Will re-upload. [{request_id}]")
                        files_to_upload_api.append((file_path, file_hash))
                        if file_hash in gemini_file_cache:
                             del gemini_file_cache[file_hash]
                else:
                    logger.info(f"Cache MISS: File {os.path.basename(file_path)} (Hash: {file_hash[:8]}...) not found in cache. Will upload. [{request_id}]")
                    files_to_upload_api.append((file_path, file_hash))
        # -----------------------------

        # --- 2. Upload necessary files to Gemini API concurrently ---
        if not files_to_upload_api:
             logger.info(f"All required files found in cache. No API uploads needed for request {request_id}.")
        else:
            update_progress(request_id, "UPLOADING", f"Uploading {len(files_to_upload_api)} files to processing service...")
            logger.info(f"Starting Gemini API uploads for {len(files_to_upload_api)} files [{request_id}]...")

            for file_path, file_hash in files_to_upload_api:
                if file_hash is None:
                    file_hash = f"no_hash_{uuid.uuid4()}"
                thread = threading.Thread(target=upload_thread_worker, args=(request_id, file_path, file_hash, uploaded_file_references, upload_errors))
                upload_threads.append(thread)
                thread.start()

            for i, thread in enumerate(upload_threads):
                thread.join()
                update_progress(request_id, "UPLOADING", f"Processing uploaded files ({i+1}/{len(files_to_upload_api)} complete)...")

            logger.info(f"All upload threads finished for [{request_id}].")

            if upload_errors:
                error_summary = "; ".join(f"{os.path.basename(k)}: {v}" for k, v in upload_errors.items())
                for fp, err in upload_errors.items():
                     logger.error(f"Upload/Processing Error for {os.path.basename(fp)} [{request_id}]: {err}")
                all_source_videos = source_media_paths.get('videos', [])
                available_source_videos = [p for p in all_source_videos if p in uploaded_file_references]
                if not available_source_videos:
                     raise ValueError(f"All source videos failed upload/processing or cache retrieval: {error_summary}")
                else:
                     logger.warning(f"Some files failed upload/processing, continuing if possible: {error_summary}")

        if not any(p in uploaded_file_references for p in source_media_paths.get('videos', [])):
             raise ValueError("API analysis requires source videos, but none are available via cache or successful upload.")

        logger.info(f"Proceeding to plan generation. API references available for {len(uploaded_file_references)} files. [{request_id}].")

        # --- 3. Generate Editing Plan ---
        plan_result = generate_editing_plan(
            request_id=request_id,
            uploaded_file_references=uploaded_file_references,
            source_media_paths=source_media_paths,
            style_description=style_description,
            sample_video_path=style_sample_path,
            target_duration=target_duration
        )

        if plan_result['status'] != 'success':
            raise ValueError(f"Failed to generate editing plan: {plan_result['message']}")

        editing_plan = plan_result['plan']
        logger.info(f"Generated Editing Plan Successfully [{request_id}]")

        # --- 4. Execute Editing Plan ---
        # --- MODIFIED: Pass the mute_all_clips_flag here ---
        execution_result = execute_editing_plan(
            request_id=request_id,
            plan=editing_plan,
            output_filename=output_filename,
            is_preview=is_preview,
            mute_all_clips=mute_all_clips_flag # Pass the flag
        )
        # -------------------------------------------------

        # --- Handle Execution Result ---
        if execution_result['status'] == 'success':
            final_output_path = execution_result['output_path']
            final_output_basename = os.path.basename(final_output_path)

            logger.info(f"Video editing successful. Output path: {final_output_path} [{request_id}]")
            logger.info(f"Preparing final result for request {request_id}. Filename: {final_output_basename}")

            with app.app_context():
                try:
                    video_url = url_for('download_file', filename=final_output_basename, _external=False)
                    logger.info(f"Download URL generated within context: {video_url} [{request_id}]")

                    result_data = {
                        'status': 'success',
                        'message': f"Video {'preview ' if is_preview else ''}generated successfully!",
                        'video_url': video_url,
                        'output_filename': final_output_basename,
                        'is_preview': is_preview,
                        'request_id': request_id
                    }
                    update_progress(request_id, "COMPLETED", f"Video generation finished successfully {'(Preview)' if is_preview else ''}.", result=result_data)
                    logger.info(f"Final success status updated for request {request_id}.")

                except Exception as url_gen_e:
                    logger.error(f"Error generating download URL within context for request {request_id}: {url_gen_e}")
                    traceback.print_exc()
                    update_progress(request_id, "FAILED", "Video generated, but failed to create download link.", error=str(url_gen_e))

        else:
            logger.error(f"Video execution plan failed for request {request_id}. Status should already be FAILED.")

    except Exception as e:
        logger.error(f"--- Unhandled Error in process_video_request Thread [{request_id}] ---")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Message: {e}")
        traceback.print_exc()
        current_status = progress_updates.get(request_id, {}).get('stage', 'UNKNOWN')
        if current_status != "FAILED":
             update_progress(request_id, "FAILED", "An unexpected error occurred during processing.", error=str(e))

    finally:
        # --- 5. Cleanup ---
        logger.info(f"Initiating cleanup for request {request_id}...")
        logger.info(f"Skipping deletion of Gemini API files for request {request_id} (Feature 1).")
        logger.info(f"Skipping deletion of local intermediate files in '{UPLOAD_FOLDER}' for request {request_id} (Feature 1).")

        if request_id in background_tasks:
            try:
                del background_tasks[request_id]
                logger.info(f"Removed background task entry for completed/failed request {request_id}")
            except KeyError:
                 logger.warning(f"Tried to remove background task entry for {request_id}, but it was already gone.")

        logger.info(f"Processing finished for request {request_id}.")


# --- Flask Routes ---

# Default style description constant
DEFAULT_STYLE_DESCRIPTION = """Project Goal: Create a fast-paced, energetic, and aesthetically beautiful promotional video showcasing the product for an Instagram home decor/kitchen channel.

Pacing: Fast, energetic, engaging.
Editing: Quick cuts.
Visuals: HIGHLY aesthetic, clean, beautiful shots ONLY. Focus on quality over quantity. Prioritize well-lit, well-composed footage. Do NOT use any mediocre or subpar shots, even if provided.

Pacing and Cuts:
Quick Cuts: Keep shot durations short (e.g., 0.5 seconds to 2 seconds max per clip).
Transitions: Mostly hard cuts will work best for this style. Avoid slow fades or complex wipes unless one specifically enhances the aesthetic (e.g., a very quick, clean wipe or maybe a smooth match-cut if the footage allows).

It's not a tutorial, it's a vibe."""


@app.route('/', methods=['GET'])
def index_get():
    now = time.time()
    ids_to_clean_progress = [rid for rid, data in list(progress_updates.items())
                             if now - data.get('timestamp', 0) > 3600 * 48]
    for rid in ids_to_clean_progress:
        if rid not in background_tasks:
            if rid in progress_updates:
                del progress_updates[rid]
                logger.info(f"Cleaned up old progress entry: {rid}")

    cleanup_expired_request_details()
    cleanup_expired_cache()

    return render_template('index.html', default_style_desc=DEFAULT_STYLE_DESCRIPTION)

@app.route('/generate', methods=['POST'])
def generate_video_post():
    global background_tasks, intermediate_files_registry, request_details_cache

    request_id = str(uuid.uuid4())
    intermediate_files_registry[request_id] = []

    try:
        # --- Form Data Validation ---
        style_description = request.form.get('style_desc', '').strip()
        target_duration_str = request.form.get('duration')
        output_filename_base = secure_filename(request.form.get('output', f'ai_edited_video_{request_id[:8]}'))
        output_filename_base = os.path.splitext(output_filename_base)[0]

        if not style_description:
            style_description = DEFAULT_STYLE_DESCRIPTION
            logger.info(f"Using default style description for request {request_id}")

        try:
            target_duration = float(target_duration_str)
            if target_duration <= 0: raise ValueError("Duration must be positive")
        except (ValueError, TypeError):
            return jsonify({'status': 'error', 'message': 'Invalid target duration. Please enter a positive number.'}), 400

        is_preview = request.form.get('generate_preview') == 'on'
        logger.info(f"Request {request_id} - Preview Mode: {is_preview}")

        # --- ADDED: Get Mute flag ---
        mute_audio_flag = request.form.get('mute_audio') == 'on'
        logger.info(f"Request {request_id} - Mute Original Audio: {mute_audio_flag}")
        # --------------------------

        output_suffix = "_preview" if is_preview else "_hq"
        output_filename = f"{output_filename_base}{output_suffix}.mp4"

        # --- File Handling ---
        saved_file_paths = {"sources": {"videos": [], "audios": [], "images": []}, "style_sample": None}
        request_files = request.files

        def save_file(file_storage, category, prefix="source"):
            if file_storage and file_storage.filename:
                if allowed_file(file_storage.filename):
                    base, ext = os.path.splitext(file_storage.filename)
                    safe_base = "".join(c if c.isalnum() or c in ('_','-','.') else '_' for c in base)
                    filename = secure_filename(f"{prefix}_{safe_base}_{request_id[:8]}{ext}")
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        file_storage.save(save_path)
                        intermediate_files_registry[request_id].append(save_path)
                        logger.info(f"Saved uploaded file [{request_id}]: {save_path}")

                        if category == "style_sample":
                            saved_file_paths["style_sample"] = save_path
                        elif category in saved_file_paths["sources"]:
                            saved_file_paths["sources"][category].append(save_path)
                        return None
                    except Exception as save_err:
                        logger.error(f"Failed to save file {filename} to {save_path}: {save_err}")
                        return f"Error saving file: {file_storage.filename}"
                else:
                    return f"Invalid file type for {category}: {file_storage.filename}"
            return None

        style_sample_error = save_file(request_files.get('style_sample'), "style_sample", prefix="style")
        if style_sample_error: return jsonify({'status': 'error', 'message': style_sample_error}), 400

        videos = request_files.getlist('videos[]')
        if not videos or all(not f.filename for f in videos):
            return jsonify({'status': 'error', 'message': 'Please upload at least one source video.'}), 400

        for video in videos:
            video_error = save_file(video, "videos")
            if video_error:
                return jsonify({'status': 'error', 'message': video_error}), 400

        if not saved_file_paths["sources"]["videos"]:
             return jsonify({'status': 'error', 'message': 'Failed to save any source videos. Check file types and permissions.'}), 400

        for audio in request_files.getlist('audios[]'):
            audio_error = save_file(audio, "audios")
            if audio_error:
                return jsonify({'status': 'error', 'message': audio_error}), 400

        for image in request_files.getlist('images[]'):
            image_error = save_file(image, "images")
            if image_error:
                return jsonify({'status': 'error', 'message': image_error}), 400

        # --- Prepare Data for Background Thread ---
        # --- MODIFIED: Added mute_audio flag ---
        form_data_for_thread = {
            'style_desc': style_description,
            'duration': target_duration,
            'output': output_filename,
            'is_preview': is_preview,
            'mute_audio': mute_audio_flag
        }
        # ---------------------------------------

        # --- Feature 3: Store Request Details ---
        request_details_cache[request_id] = {
            'form_data': form_data_for_thread.copy(),
            'file_paths': saved_file_paths.copy(),
            'timestamp': time.time()
        }
        logger.info(f"Stored request details for potential HQ generation. ID: {request_id}")
        # --------------------------------------

        # --- Start Background Thread ---
        update_progress(request_id, "RECEIVED", "Request received. Initializing processing...")
        thread = threading.Thread(target=process_video_request, args=(request_id, form_data_for_thread, saved_file_paths, app))
        background_tasks[request_id] = thread
        thread.start()

        logger.info(f"Started background processing thread for request ID: {request_id}")

        # --- Return Immediate Response ---
        return jsonify({
            'status': 'processing_started',
            'message': 'Video generation process started. You can monitor the progress.',
            'request_id': request_id
        })

    except Exception as e:
        logger.error(f"--- Error in /generate endpoint before starting thread [{request_id}] ---")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f"An internal server error occurred during setup: {e}"}), 500

# --- Feature 3: New Route for High-Quality Generation ---
@app.route('/generate-hq/<preview_request_id>', methods=['POST'])
def generate_high_quality_video(preview_request_id):
    global background_tasks, request_details_cache

    logger.info(f"Received request to generate High Quality video based on preview ID: {preview_request_id}")

    original_details = request_details_cache.get(preview_request_id)
    if not original_details:
        logger.error(f"Original request details not found for preview ID: {preview_request_id}")
        return jsonify({'status': 'error', 'message': 'Original request details not found. Cannot generate high-quality version.'}), 404

    hq_request_id = str(uuid.uuid4())
    logger.info(f"Generating HQ video with new request ID: {hq_request_id}")

    hq_form_data = original_details['form_data'].copy()
    hq_form_data['is_preview'] = False # Set to HQ mode

    base_output_name = os.path.splitext(hq_form_data['output'])[0]
    if base_output_name.endswith('_preview'):
        base_output_name = base_output_name[:-len('_preview')]
    hq_form_data['output'] = f"{base_output_name}_hq.mp4"

    hq_file_paths = original_details['file_paths'].copy()

    request_details_cache[hq_request_id] = {
        'form_data': hq_form_data.copy(),
        'file_paths': hq_file_paths.copy(),
        'timestamp': time.time(),
        'based_on_preview_id': preview_request_id
    }

    update_progress(hq_request_id, "RECEIVED", "High-Quality generation request received. Initializing...")
    thread = threading.Thread(target=process_video_request, args=(hq_request_id, hq_form_data, hq_file_paths, app))
    background_tasks[hq_request_id] = thread
    thread.start()

    logger.info(f"Started background processing thread for HQ request ID: {hq_request_id}")

    return jsonify({
        'status': 'processing_started',
        'message': 'High-Quality video generation process started.',
        'request_id': hq_request_id
    })
# ----------------------------------------------------

@app.route('/progress/<request_id>', methods=['GET'])
def get_progress(request_id):
    """Endpoint for the client to poll for progress updates."""
    progress_data = progress_updates.get(request_id)

    if not progress_data:
        return jsonify({"stage": "UNKNOWN", "message": "Request ID not found or expired.", "error": None, "result": None}), 404

    return jsonify(progress_data)


@app.route('/output/<path:filename>')
def download_file(filename):
    """Serves the final generated video file."""
    safe_filename = secure_filename(filename)
    if safe_filename != filename or '/' in filename or '\\' in filename:
        logger.warning(f"Attempt to access potentially unsafe path rejected: {filename}")
        return "Invalid filename", 400

    file_path = os.path.join(app.config['FINAL_OUTPUT_FOLDER'], safe_filename)
    logger.info(f"Attempting to send file: {file_path}")

    if not os.path.exists(file_path):
         logger.error(f"Download request: File not found at {file_path}")
         return "File not found", 404

    response = send_file(file_path, as_attachment=False)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)