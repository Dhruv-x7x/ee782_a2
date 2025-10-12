#!/usr/bin/env python3
"""
main.py — AI Guard Agent (Corrected Version)

Fixed Issues:
- Proper threading locks for audio resource management
- webrtcvad for Voice Activity Detection
- MediaPipe for faster face detection
- Fixed enrollment process (only works when disarmed)
- Collective escalation for unknown faces
- Comprehensive error handling and recovery
- Resource cleanup handlers
- Removed "add activation phrase" feature

Requirements: Python 3.11.9, Windows
"""

import os
import time
import queue
import threading
import traceback
import atexit
import signal
import sys
from datetime import datetime

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import pyttsx3
import webrtcvad
import mediapipe as mp
from deepface import DeepFace

# --------------------------
# Configuration
# --------------------------
INPUT_DEVICE_INDEX = 4
OUTPUT_DEVICE_INDEX = None

ASR_RECORD_SECONDS = 2.0      # Increased for better transcription
ENROLL_NAME_RECORD_SECONDS = 2.0

ACTIVATION_PHRASES_FILE = "activation_phrases.txt"
DEFAULT_ACTIVATION_PHRASES = [
    "guard mode on",
    "guard on",
    "guardmode on",
    "guard mode activate",
    "cardboard on",
    "card on",
    "god on",
    "guard activate",
    "mode on",
    "card mode on"
]

DEACTIVATION_PHRASE = "guard mode off"
ENROLL_COMMAND = "enroll"

ENROLL_DIR = "enrolled"
os.makedirs(ENROLL_DIR, exist_ok=True)

CAM_INDEX = 0
ENROLL_PHOTOS = 5
COSINE_THRESHOLD = 0.35
WARNING_AFTER = 3.0
ESCALATE_AFTER = 8.0
ESCALATION_COOLDOWN = 30.0  # Seconds between repeated escalation messages

FACE_PROCESS_INTERVAL = 1.5  # Process faces every 1.5 seconds
VAD_AGGRESSIVENESS = 2  # Reduced from 3 to 2 for better balance
VAD_SPEECH_THRESHOLD = 0.4  # Increased from 0.3 - require 40% speech frames

# --------------------------
# Global locks and state
# --------------------------
audio_lock = threading.Lock()  # Proper lock for audio resources
tts_active = threading.Event()
cleanup_files = []  # Track temp files for cleanup

# --------------------------
# Cleanup handler
# --------------------------
def cleanup_resources():
    """Clean up resources on exit"""
    print("\n[CLEANUP] Cleaning up resources...")
    for fpath in cleanup_files:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
        except Exception as e:
            print(f"[CLEANUP] Failed to remove {fpath}: {e}")
    cv2.destroyAllWindows()

atexit.register(cleanup_resources)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n[SIGNAL] Interrupt received, shutting down...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --------------------------
# Load activation phrases
# --------------------------
def load_activation_phrases():
    """Load activation phrases from file"""
    if not os.path.exists(ACTIVATION_PHRASES_FILE):
        with open(ACTIVATION_PHRASES_FILE, "w", encoding="utf-8") as f:
            for p in DEFAULT_ACTIVATION_PHRASES:
                f.write(p.strip() + "\n")
        return list(DEFAULT_ACTIVATION_PHRASES)
    else:
        with open(ACTIVATION_PHRASES_FILE, "r", encoding="utf-8") as f:
            phrases = [line.strip().lower() for line in f.readlines() if line.strip()]
        if not phrases:
            phrases = list(DEFAULT_ACTIVATION_PHRASES)
        return phrases

activation_phrases = load_activation_phrases()

# --------------------------
# Audio device setup
# --------------------------
def setup_audio_devices():
    """Setup and validate audio devices"""
    try:
        devices = sd.query_devices()
        print(f"\n[AUDIO] Available devices:")
        for i, dev in enumerate(devices):
            print(f"  [{i}] {dev['name']}")
        
        # Validate input device
        input_dev = sd.query_devices(INPUT_DEVICE_INDEX)
        if input_dev['max_input_channels'] < 1:
            print(f"[AUDIO] WARNING: Device {INPUT_DEVICE_INDEX} has no input channels!")
            return False
        
        sd.default.device = (INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX)
        print(f"[AUDIO] Set default input to device {INPUT_DEVICE_INDEX}: {input_dev['name']}")
        return True
    except Exception as e:
        print(f"[AUDIO] Error setting up audio devices: {e}")
        traceback.print_exc()
        return False

# --------------------------
# TTS with proper locking
# --------------------------
tts_engine = None

def init_tts():
    """Initialize TTS engine with error handling"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        return True
    except Exception as e:
        print(f"[TTS] Failed to initialize: {e}")
        return False

def say(text):
    """Thread-safe TTS with audio lock"""
    global tts_engine
    if tts_engine is None:
        print(f"[TTS] Engine not initialized. Text: {text}")
        return
    
    with audio_lock:
        tts_active.set()
        try:
            print(f"[TTS] {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            tts_active.clear()
            time.sleep(0.2)  # Small delay to ensure audio output completes

# --------------------------
# VAD-enhanced recorder
# --------------------------
def record_wav_with_vad(device_index, duration, vad_mode=VAD_AGGRESSIVENESS):
    """Record audio with Voice Activity Detection"""
    try:
        dev = sd.query_devices(device_index)
        # Get device sample rate
        device_samplerate = int(dev.get('default_samplerate', 16000))
        target_samplerate = 16000  # VAD requires 16kHz
        channels = 1
        
        fname = f"tmp_asr_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}.wav"
        cleanup_files.append(fname)
        
        with audio_lock:
            rec = sd.rec(
                int(duration * device_samplerate),
                samplerate=device_samplerate,
                channels=channels,
                dtype='int16',
                device=device_index
            )
            sd.wait()
        
        rec = np.asarray(rec).flatten()  # Ensure 1D array
        
        # Resample to 16kHz if needed (VAD requirement)
        if device_samplerate != target_samplerate:
            try:
                from scipy import signal
                num_samples = int(len(rec) * target_samplerate / device_samplerate)
                rec = signal.resample(rec, num_samples).astype('int16')
            except ImportError:
                print("[ASR] Warning: scipy not available, using original sample rate")
                target_samplerate = device_samplerate
        
        # Voice Activity Detection
        vad = webrtcvad.Vad(vad_mode)
        frame_duration = 30  # ms
        frame_size = int(target_samplerate * frame_duration / 1000)
        
        voiced_frames = 0
        total_frames = 0
        
        for i in range(0, len(rec) - frame_size, frame_size):
            frame = rec[i:i + frame_size].tobytes()
            total_frames += 1
            try:
                if vad.is_speech(frame, target_samplerate):
                    voiced_frames += 1
            except:
                pass
        
        # Calculate speech ratio
        speech_ratio = voiced_frames / total_frames if total_frames > 0 else 0
        
        # Save only if sufficient speech detected
        if speech_ratio > VAD_SPEECH_THRESHOLD:  # Configurable threshold
            sf.write(fname, rec, target_samplerate)
            return fname, speech_ratio
        else:
            if fname in cleanup_files:
                cleanup_files.remove(fname)
            return None, speech_ratio
            
    except Exception as e:
        print(f"[ASR] Recording failed: {e}")
        traceback.print_exc()
        return None, 0.0

# --------------------------
# ASR transcription
# --------------------------
def transcribe_wav_file(path):
    """Transcribe audio file with error handling"""
    if not path or not os.path.exists(path):
        return ""
    
    r = sr.Recognizer()
    try:
        with sr.AudioFile(path) as src:
            r.adjust_for_ambient_noise(src, duration=0.2)
            aud = r.record(src)
        txt = r.recognize_google(aud)
        return txt.lower().strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"[ASR] Google API error: {e}")
        return ""
    except Exception as e:
        print(f"[ASR] Transcription error: {e}")
        return ""

# --------------------------
# MediaPipe face detection
# --------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces_mediapipe(frame, confidence=0.5):
    """Detect faces using MediaPipe (faster than Haar Cascade)"""
    faces = []
    try:
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence
        ) as face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure bbox is within frame
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    faces.append((x, y, width, height))
    except Exception as e:
        print(f"[MEDIAPIPE] Detection error: {e}")
    
    return faces

# --------------------------
# DeepFace embedding functions
# --------------------------
def get_embedding_from_image(img_bgr, model_name="ArcFace", retry_count=2):
    """Get face embedding with retry logic"""
    for attempt in range(retry_count):
        try:
            rep = DeepFace.represent(
                img_path=img_bgr,
                model_name=model_name,
                enforce_detection=False,
                detector_backend='skip'  # We already detected face with MediaPipe
            )
            if isinstance(rep, list):
                rep = rep[0]
            emb = np.array(rep["embedding"], dtype=np.float32)
            return emb
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"[DEEPFACE] Retry {attempt + 1}/{retry_count}")
                time.sleep(0.1)
            else:
                print(f"[DEEPFACE] Embedding failed after {retry_count} attempts: {e}")
                return None
    return None

def cosine_distance(a, b):
    """Calculate cosine distance between embeddings"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    sim = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - sim

# --------------------------
# Enrollment persistence
# --------------------------
def save_enrollment(name, embeddings):
    """Save enrollment data"""
    try:
        path = os.path.join(ENROLL_DIR, f"{name}.npz")
        np.savez_compressed(
            path,
            embeddings=np.asarray(embeddings),
            created_at=datetime.utcnow().isoformat()
        )
        print(f"[ENROLL] Saved enrollment for '{name}' -> {path}")
        return True
    except Exception as e:
        print(f"[ENROLL] Failed to save enrollment: {e}")
        return False

def load_enrolled_db():
    """Load all enrolled faces"""
    db = {}
    try:
        for fn in os.listdir(ENROLL_DIR):
            if fn.lower().endswith(".npz"):
                try:
                    data = np.load(os.path.join(ENROLL_DIR, fn), allow_pickle=True)
                    emb = np.asarray(data["embeddings"])
                    name = os.path.splitext(fn)[0]
                    db[name] = emb
                    print(f"[ENROLL] Loaded {name}: {len(emb)} embeddings")
                except Exception as e:
                    print(f"[ENROLL] Failed to load {fn}: {e}")
    except Exception as e:
        print(f"[ENROLL] Error loading database: {e}")
    
    print(f"[ENROLL] Total enrolled: {list(db.keys())}")
    return db

def match_embedding(db, emb):
    """Match embedding against enrolled database"""
    if not db or emb is None:
        return None, 1.0
    
    best_name, best_score = None, 1.0
    for name, embs in db.items():
        dists = [cosine_distance(emb, e) for e in embs]
        min_dist = float(np.min(dists))
        if min_dist < best_score:
            best_score = min_dist
            best_name = name
    
    return best_name, best_score

# --------------------------
# Command matching with fuzzy logic
# --------------------------
def matches_activation_command(heard_text, activation_phrases):
    """
    Flexible matching for activation commands.
    Allows partial matches and common misheard words.
    """
    if not heard_text:
        return False, None
    
    heard = heard_text.lower().strip()
    
    # Direct substring match
    for phrase in activation_phrases:
        if phrase in heard:
            return True, phrase
    
    # Check for key activation words
    activation_keywords = ['guard', 'god', 'card', 'cardboard', 'got']
    action_keywords = ['on', 'activate', 'start', 'begin', 'mode']
    
    heard_words = heard.split()
    
    # If we hear an activation keyword + action keyword, activate
    has_activation = any(kw in heard for kw in activation_keywords)
    has_action = any(kw in heard for kw in action_keywords)
    
    if has_activation and has_action:
        return True, "fuzzy_match"
    
    # Single word commands
    if heard in ['guard', 'god', 'start', 'activate']:
        return True, "single_word"
    
    return False, None

def matches_command(heard_text, command_phrase):
    """Check if heard text matches a specific command"""
    if not heard_text:
        return False
    heard = heard_text.lower().strip()
    command = command_phrase.lower().strip()
    
    # Exact substring match
    if command in heard:
        return True
    
    # For "enroll" - allow "enrol", "and roll", "in roll"
    if command == "enroll":
        enroll_variants = ['enroll', 'enrol', 'and roll', 'in roll', 'roll']
        if any(variant in heard for variant in enroll_variants):
            return True
    
    # For deactivation - allow "off", "stop", "deactivate"
    if "off" in command or "deactivate" in command:
        if any(word in heard for word in ['off', 'stop', 'deactivate', 'end', 'disable']):
            return True
    
    # Check word overlap (flexible matching)
    heard_words = set(heard.split())
    command_words = set(command.split())
    overlap = len(heard_words & command_words)
    
    # If most command words are present, consider it a match
    if len(command_words) > 0 and overlap / len(command_words) >= 0.6:
        return True
    
    return False

# --------------------------
# ASR listener thread
# --------------------------
class ASRListener(threading.Thread):
    def __init__(self, out_q, stop_event):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event
        self.running = True

    def run(self):
        print("[ASR] Listener started with VAD")
        consecutive_silence = 0
        
        while not self.stop_event.is_set() and self.running:
            try:
                # Wait if TTS is active
                if tts_active.is_set():
                    time.sleep(0.1)
                    continue
                
                fname, speech_ratio = record_wav_with_vad(
                    INPUT_DEVICE_INDEX,
                    ASR_RECORD_SECONDS
                )
                
                if not fname:
                    consecutive_silence += 1
                    if consecutive_silence % 10 == 0:
                        print(f"[ASR] Listening... (silence count: {consecutive_silence})")
                    time.sleep(0.05)
                    continue
                
                consecutive_silence = 0
                # Don't print speech detection ratio every time - too noisy
                
                txt = transcribe_wav_file(fname)
                
                # Cleanup temp file
                try:
                    if fname in cleanup_files:
                        cleanup_files.remove(fname)
                    if os.path.exists(fname):
                        os.remove(fname)
                except:
                    pass
                
                if txt:
                    print(f"[ASR] >>> HEARD: \"{txt}\"")  # Clear output of what was heard
                    self.q.put(txt)
                else:
                    print(f"[ASR] (Speech detected but couldn't transcribe)")
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"[ASR] Error in listener loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)

    def stop(self):
        self.running = False

# --------------------------
# Enrollment process (FIXED)
# --------------------------
def perform_enrollment(cam, enrolled_db):
    """Voice-guided enrollment with proper face validation"""
    say("Starting enrollment. Please say the name after the beep.")
    time.sleep(0.5)
    say("Beep.")
    
    # Record name
    name_file, _ = record_wav_with_vad(INPUT_DEVICE_INDEX, ENROLL_NAME_RECORD_SECONDS)
    if not name_file:
        say("Could not record name. Enrollment cancelled.")
        return enrolled_db
    
    name_text = transcribe_wav_file(name_file)
    
    # Cleanup
    try:
        if name_file in cleanup_files:
            cleanup_files.remove(name_file)
        if os.path.exists(name_file):
            os.remove(name_file)
    except:
        pass
    
    if not name_text:
        say("I could not understand the name. Enrollment cancelled.")
        return enrolled_db
    
    spoken_name = name_text.split()[0].lower()
    print(f"[ENROLL] Enrolling: '{spoken_name}'")
    say(f"Enrolling {spoken_name}. I will capture your face from different angles.")
    
    # Pose prompts
    pose_prompts = [
        "Look straight at the camera",
        "Turn your head slightly to the left",
        "Turn your head slightly to the right",
        "Tilt your head slightly up",
        "Tilt your head slightly down"
    ]
    
    captures = []
    say("Get ready. Position your face in the green box.")
    time.sleep(1.0)
    
    for idx, prompt in enumerate(pose_prompts[:ENROLL_PHOTOS]):
        say(prompt)
        time.sleep(1.5)
        
        best_crop = None
        best_area = 0
        
        # Try to capture for 1.5 seconds
        start_time = time.time()
        while time.time() - start_time < 1.5:
            ret, frame = cam.read()
            if not ret:
                continue
            
            faces = detect_faces_mediapipe(frame, confidence=0.7)
            
            # Display guidance
            display = frame.copy()
            h, w = display.shape[:2]
            cv2.putText(
                display,
                f"Enrollment: {idx + 1}/{ENROLL_PHOTOS}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display,
                prompt,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Draw target box
            cx1, cy1 = int(w * 0.3), int(h * 0.2)
            cx2, cy2 = int(w * 0.7), int(h * 0.8)
            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
            
            # Draw detected faces
            for (x, y, fw, fh) in faces:
                cv2.rectangle(display, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
                area = fw * fh
                if area > best_area:
                    best_area = area
                    best_crop = frame[y:y + fh, x:x + fw].copy()
            
            cv2.imshow("Enrollment (ESC to cancel)", display)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                say("Enrollment cancelled.")
                cv2.destroyAllWindows()
                return enrolled_db
        
        if best_crop is not None and best_crop.size > 0:
            captures.append(best_crop)
            say(f"Captured pose {idx + 1}")
            print(f"[ENROLL] Captured pose {idx + 1}, area={best_area}")
        else:
            say(f"No face detected for this pose. Skipping.")
            print(f"[ENROLL] No face for pose {idx + 1}")
    
    cv2.destroyAllWindows()
    
    if len(captures) < 2:
        say("Enrollment failed. Not enough valid captures. Please try again.")
        return enrolled_db
    
    say("Computing face embeddings. Please wait.")
    embeddings = []
    
    for i, img in enumerate(captures):
        print(f"[ENROLL] Computing embedding {i + 1}/{len(captures)}...")
        emb = get_embedding_from_image(img)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"[ENROLL] Failed to compute embedding for capture {i + 1}")
    
    if len(embeddings) < 2:
        say("Enrollment failed. Could not extract face features. Please try again.")
        return enrolled_db
    
    # Validate consistency (optional but recommended)
    if len(embeddings) >= 2:
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = cosine_distance(embeddings[i], embeddings[j])
                distances.append(dist)
        
        avg_dist = np.mean(distances)
        print(f"[ENROLL] Average intra-person distance: {avg_dist:.3f}")
        
        if avg_dist > 0.5:
            say("Warning: Face captures show high variation. Enrollment may not be reliable.")
    
    # Save enrollment
    if save_enrollment(spoken_name, embeddings):
        enrolled_db = load_enrolled_db()
        say(f"Enrollment complete for {spoken_name}. {len(embeddings)} face embeddings saved.")
        print(f"[ENROLL] Success: '{spoken_name}' enrolled with {len(embeddings)} embeddings")
    else:
        say("Failed to save enrollment data.")
    
    return enrolled_db

# --------------------------
# Main guard loop
# --------------------------
def guard_main_loop():
    global activation_phrases
    
    # Initialize systems
    print("\n[INIT] Initializing AI Guard Agent...")
    
    if not setup_audio_devices():
        print("[ERROR] Audio device setup failed. Exiting.")
        return
    
    if not init_tts():
        print("[ERROR] TTS initialization failed. Exiting.")
        return
    
    activation_phrases = load_activation_phrases()
    print(f"[INIT] Activation phrases: {activation_phrases}")
    
    enrolled_db = load_enrolled_db()
    
    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Exiting.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Start ASR listener
    cmd_q = queue.Queue()
    stop_event = threading.Event()
    asr = ASRListener(cmd_q, stop_event)
    asr.start()
    
    # State variables
    armed = False
    state = "IDLE"
    unrec_start = None
    last_escalation_time = 0
    last_face_process = 0.0
    
    say("AI Guard Agent ready. Say 'guard mode on' to activate monitoring.")
    print("\n[INFO] Press ESC in camera window to exit.")
    print("[INFO] Say 'enroll' to add a trusted person (only when disarmed).")
    
    try:
        while True:
            # Process commands
            while not cmd_q.empty():
                txt = cmd_q.get_nowait()
                print(f"[CMD] Processing: \"{txt}\"")
                
                # Enrollment command (only when disarmed)
                if matches_command(txt, ENROLL_COMMAND):
                    if armed:
                        say("Cannot enroll while guard mode is active. Please disarm first.")
                        print("[CMD] Enrollment blocked: guard is armed")
                    else:
                        say("Enrollment command received.")
                        enrolled_db = perform_enrollment(cap, enrolled_db)
                    continue
                
                # Deactivation
                if matches_command(txt, DEACTIVATION_PHRASE) or matches_command(txt, "off"):
                    if armed:
                        armed = False
                        state = "IDLE"
                        unrec_start = None
                        last_escalation_time = 0
                        say("Guard mode deactivated.")
                        print("[STATE] DISARMED")
                    else:
                        print("[CMD] Already disarmed")
                    continue
                
                # Activation - use new fuzzy matching
                is_activation, matched_phrase = matches_activation_command(txt, activation_phrases)
                if is_activation:
                    if not armed:
                        armed = True
                        state = "ARMED"
                        unrec_start = None
                        last_escalation_time = 0
                        say("Guard mode activated. Monitoring area.")
                        print(f"[STATE] ARMED (matched: '{matched_phrase}' from input: '{txt}')")
                    else:
                        print("[CMD] Already armed")
                    continue
                
                # If no command matched, just log it
                print(f"[CMD] No command matched for: \"{txt}\"")
            
            # Camera capture
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Display frame with status
            display = frame.copy()
            h, w = display.shape[:2]
            
            if armed:
                cv2.putText(
                    display,
                    "GUARD MODE: ARMED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    display,
                    "GUARD MODE: DISARMED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 160, 0),
                    2
                )
            
            # Face detection and recognition (only when armed)
            current_time = time.time()
            if armed and (current_time - last_face_process) > FACE_PROCESS_INTERVAL:
                last_face_process = current_time
                
                faces = detect_faces_mediapipe(frame, confidence=0.5)
                
                if len(faces) == 0:
                    # No faces detected - reset state
                    if unrec_start is not None:
                        print("[GUARD] No faces detected - resetting")
                    unrec_start = None
                    state = "ARMED"
                else:
                    # Process detected faces
                    any_unrecognized = False
                    
                    for (x, y, fw, fh) in faces:
                        face_crop = frame[y:y + fh, x:x + fw].copy()
                        
                        if face_crop.size == 0:
                            continue
                        
                        # Get embedding
                        emb = get_embedding_from_image(face_crop)
                        
                        label = "unknown"
                        dist = 1.0
                        
                        if emb is not None and len(enrolled_db) > 0:
                            best_name, best_dist = match_embedding(enrolled_db, emb)
                            dist = best_dist
                            
                            if best_name and best_dist <= COSINE_THRESHOLD:
                                label = best_name
                            else:
                                label = "unknown"
                        else:
                            label = "unknown"
                        
                        # Draw bounding box
                        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                        cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 2)
                        cv2.putText(
                            display,
                            f"{label} ({dist:.2f})",
                            (x, max(20, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1
                        )
                        
                        if label == "unknown":
                            any_unrecognized = True
                    
                    # Collective escalation logic
                    if any_unrecognized:
                        if unrec_start is None:
                            unrec_start = current_time
                            state = "UNRECOGNIZED_DETECTED"
                            say("Hello. Please identify yourself.")
                            print("[ESCALATE] Initial greeting")
                            last_escalation_time = current_time
                        else:
                            elapsed = current_time - unrec_start
                            time_since_last_msg = current_time - last_escalation_time
                            
                            # Warning phase
                            if state == "UNRECOGNIZED_DETECTED" and elapsed > WARNING_AFTER:
                                if time_since_last_msg > ESCALATION_COOLDOWN:
                                    state = "WARNING"
                                    say("You are not authorized. Please leave immediately.")
                                    print("[ESCALATE] Warning issued")
                                    last_escalation_time = current_time
                            
                            # Escalation phase
                            if state == "WARNING" and elapsed > ESCALATE_AFTER:
                                if time_since_last_msg > ESCALATION_COOLDOWN:
                                    state = "ESCALATED"
                                    say("This is your final warning. This area is monitored. Leave now.")
                                    print("[ESCALATE] Final warning")
                                    last_escalation_time = current_time
                            
                            # Continue escalated warnings periodically
                            if state == "ESCALATED" and time_since_last_msg > ESCALATION_COOLDOWN:
                                say("Unauthorized presence detected. Authorities will be notified.")
                                print("[ESCALATE] Repeated warning")
                                last_escalation_time = current_time
                    else:
                        # All faces recognized - reset
                        if unrec_start is not None:
                            print("[GUARD] All faces recognized - resetting")
                            unrec_start = None
                            state = "ARMED"
            
            # Display state info
            if state != "IDLE" and state != "ARMED":
                cv2.putText(
                    display,
                    f"State: {state}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1
                )
            
            cv2.imshow("AI Guard - Press ESC to exit", display)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                say("Shutting down. Goodbye.")
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
        traceback.print_exc()
    finally:
        print("[MAIN] Cleaning up...")
        stop_event.set()
        asr.stop()
        asr.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete")

# --------------------------
# Entry point
# --------------------------
if __name__ == "__main__":
    try:
        guard_main_loop()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
    finally:
        cleanup_resources()