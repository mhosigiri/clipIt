from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid
import threading
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scenedetect import detect, ContentDetector
import mediapipe as mp
import google.generativeai as genai
import json
import logging
from memory_store import MemoryStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scene_cache = {}
cache_lock = threading.Lock()

app = Flask(__name__)
CORS(app)

# Gemini configuration
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
UPLOAD_DIR = "uploads"
SNIPPET_DIR = "snippets"
FEEDBACK_LOG = "feedback_log.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SNIPPET_DIR, exist_ok=True)

# Initialize memory store
memory_store = MemoryStore("clip_memory.json")

# Helper to save feedback to JSON file
def save_feedback(data):
    try:
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
        feedbacks.append(data)
        with open(FEEDBACK_LOG, "w") as f:
            json.dump(feedbacks, f, indent=2)
    except Exception as e:
        print("Error saving feedback:", e)

# Initialize MediaPipe Face Detection
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5
)

def validate_video(video_path: str):
    """Validate video file before processing."""
    if not os.path.exists(video_path):
        raise ValueError("Video file does not exist")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open video file - invalid format or corrupted")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_count == 0:
        cap.release()
        raise ValueError("Video has no frames")
    
    if fps <= 0:
        cap.release()
        raise ValueError("Invalid video frame rate")
    
    # Check if we can read at least the first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise ValueError("Cannot read video frames")
    
    cap.release()
    logger.info(f"Video validation passed: {frame_count} frames, {fps} fps")
    return frame_count, fps

def analyze_video(video_path: str):
    """Analyze video to find interesting scenes using face detection and motion analysis."""
    logger.info(f"Starting video analysis for: {video_path}")
    
    # Validate video first
    frame_count, fps = validate_video(video_path)
    
    scenes = []
    cap = cv2.VideoCapture(video_path)
    total_frames = frame_count
    
    # Detect scenes using PySceneDetect
    logger.info("Detecting scenes with PySceneDetect...")
    try:
        scene_list = detect(video_path, ContentDetector())
        logger.info(f"Found {len(scene_list)} scenes")
    except Exception as e:
        logger.error(f"Scene detection failed: {str(e)}")
        cap.release()
        raise ValueError(f"Scene detection failed: {str(e)}")
    
    for scene in scene_list:
        start_frame = int(scene[0].frame_num)
        end_frame = int(scene[1].frame_num)
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Skip very short or very long scenes
        if end_time - start_time < 3 or end_time - start_time > 60:
            continue
            
        # Analyze frames in the scene
        logger.info(f"Analyzing scene {len(scenes)+1}: {start_time:.2f}s to {end_time:.2f}s")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        face_count = 0
        motion_score = 0
        prev_frame = None
        processed_frames = 0
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx}")
                break
            
            processed_frames += 1
            
            try:
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection with error handling
                try:
                    results = face_detection.process(frame_rgb)
                    if results and results.detections:
                        face_count += len(results.detections)
                except Exception as e:
                    logger.warning(f"Face detection failed for frame {frame_idx}: {str(e)}")
                    # Continue without face detection for this frame
                
                # Motion detection
                if prev_frame is not None:
                    try:
                        diff = cv2.absdiff(frame, prev_frame)
                        motion_score += np.mean(diff)
                    except Exception as e:
                        logger.warning(f"Motion detection failed for frame {frame_idx}: {str(e)}")
                
                prev_frame = frame.copy()
                
            except Exception as e:
                logger.warning(f"Frame processing failed for frame {frame_idx}: {str(e)}")
                continue
        
        logger.info(f"Scene analysis complete: {processed_frames} frames processed, {face_count} faces detected")
        
        # Calculate scene score
        scene_score = (face_count * 0.6 + motion_score * 0.4) / (end_frame - start_frame)
        scenes.append({
            'start_time': start_time,
            'end_time': end_time,
            'score': scene_score
        })
    
    cap.release()
    return sorted(scenes, key=lambda x: x['score'], reverse=True)

def extract_clip(video_path: str, start_time: float, end_time: float, output_path: str):
    """Extract a clip from the video using moviepy."""
    logger.info(f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s")
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Validate time ranges
        if start_time < 0 or end_time > duration or start_time >= end_time:
            video.close()
            raise ValueError(f"Invalid time range: {start_time:.2f}s to {end_time:.2f}s (video duration: {duration:.2f}s)")
        
        clip = video.subclip(start_time, end_time)
        
        # Extract with error handling
        clip.write_videofile(
            output_path, 
            codec='libx264', 
            audio_codec='aac',
            verbose=False,
            logger=None  # Suppress moviepy logs
        )
        
        video.close()
        clip.close()
        
        logger.info(f"Clip extraction completed: {output_path}")
        
    except Exception as e:
        logger.error(f"Clip extraction failed: {str(e)}")
        raise ValueError(f"Failed to extract clip: {str(e)}")

# Gemini helper function
def get_best_scene_with_gemini(prompt: str, scenes: list) -> dict:
    model = genai.GenerativeModel("gemini-pro")
    top_scenes = scenes[:5]  # Send top 5
    scene_descriptions = "\n".join([
        f"- Scene {i+1}: {s['start_time']:.2f}s to {s['end_time']:.2f}s, score {s['score']:.2f}"
        for i, s in enumerate(top_scenes)
    ])
    prompt_text = f"""A user uploaded a video and asked: "{prompt}".
Here are the top scenes extracted:
{scene_descriptions}

Based on the prompt, which scene (by number) is the best match?
Just return a number (1-5)."""
    try:
        response = model.generate_content(prompt_text)
        choice = int("".join(filter(str.isdigit, response.text.strip())))
        return top_scenes[choice - 1] if 1 <= choice <= len(top_scenes) else top_scenes[0]
    except:
        return top_scenes[0]

@app.route("/analyze", methods=["POST"])
def analyze_video_endpoint():
    """API endpoint to analyze video and extract the most interesting clip."""
    logger.info("Received video analysis request")
    
    vid = request.files.get("video")
    prompt = request.form.get("prompt", "")
    
    if not vid:
        logger.error("No video file provided")
        return jsonify({"error": "Missing video file"}), 400
    
    if not vid.filename:
        logger.error("Video file has no filename")
        return jsonify({"error": "Invalid video file"}), 400
        
    # Save uploaded video
    video_id = str(uuid.uuid4())
    video_ext = os.path.splitext(vid.filename)[1].lower()
    
    # Validate video file extension
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v']
    if video_ext not in allowed_extensions:
        logger.error(f"Unsupported video format: {video_ext}")
        return jsonify({"error": f"Unsupported video format. Allowed: {', '.join(allowed_extensions)}"}), 400
    
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}{video_ext}")
    logger.info(f"Saving video as: {video_path}")
    
    try:
        vid.save(video_path)
        file_size = os.path.getsize(video_path)
        logger.info(f"Video saved successfully: {file_size} bytes")
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        return jsonify({"error": "Failed to save video file"}), 500
    
    try:
        # Analyze video to find interesting scenes
        logger.info("Starting video scene analysis...")
        scenes = analyze_video(video_path)
        
        if not scenes:
            logger.warning("No suitable scenes found in the video")
            return jsonify({"error": "No suitable scenes found in the video. Try a different video or check if it contains clear scenes."}), 400
        
        logger.info(f"Found {len(scenes)} suitable scenes")
        
        # Get the best scene
        logger.info(f"Selecting best scene using prompt: '{prompt}'")
        best_scene = get_best_scene_with_gemini(prompt, scenes)
        logger.info(f"Selected scene: {best_scene['start_time']:.2f}s to {best_scene['end_time']:.2f}s")
        
        # Extract the clip
        snippet_filename = f"{video_id}_snippet.mp4"
        snippet_path = os.path.join(SNIPPET_DIR, snippet_filename)
        
        logger.info(f"Extracting clip to: {snippet_path}")
        extract_clip(
            video_path,
            best_scene['start_time'],
            best_scene['end_time'],
            snippet_path
        )
        
        if not os.path.exists(snippet_path):
            logger.error("Clip extraction failed - output file not created")
            return jsonify({"error": "Failed to extract video clip"}), 500
        
        clip_size = os.path.getsize(snippet_path)
        logger.info(f"Clip extracted successfully: {clip_size} bytes")
        
        # Store memory
        memory_store.add_memory(video_id, {
            "prompt": prompt,
            "start_time": best_scene['start_time'],
            "end_time": best_scene['end_time'],
            "snippet_path": snippet_path
        })
        
        logger.info("Sending extracted clip to client")
        return send_file(snippet_path, mimetype="video/mp4")
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded video
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

# Feedback endpoint
@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect user feedback after snippet review."""
    data = request.get_json()
    required_keys = {"video_id", "prompt", "satisfied"}
    if not data or not required_keys.issubset(data):
        return jsonify({"error": "Missing required feedback fields"}), 400
    save_feedback(data)
    return jsonify({"status": "Feedback received"})

# Stats endpoint
@app.route("/stats", methods=["GET"])
def stats():
    """Return basic stats about the app."""
    try:
        # Count snippets generated
        snippet_count = len([f for f in os.listdir(SNIPPET_DIR) if f.endswith('.mp4')])
        
        # Count feedback
        feedback_count = 0
        positive_feedback = 0
        if os.path.exists(FEEDBACK_LOG):
            with open(FEEDBACK_LOG, "r") as f:
                feedbacks = json.load(f)
                feedback_count = len(feedbacks)
                positive_feedback = sum(1 for f in feedbacks if f.get("satisfied", False))
        
        return jsonify({
            "total_clips": snippet_count,
            "feedback_count": feedback_count,
            "positive_feedback": positive_feedback,
            "satisfaction_rate": positive_feedback / feedback_count if feedback_count > 0 else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local development only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)
else:
    # For production
    gunicorn_app = app