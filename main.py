#!/usr/bin/env python3

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

from concurrent.futures import ThreadPoolExecutor
import contextlib
# --------------------------
# Configuration
# --------------------------
INPUT_DEVICE_INDEX = 4
OUTPUT_DEVICE_INDEX = None

ASR_RECORD_SECONDS = 2.0
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
# near other globals
asr_pause = threading.Event()   # when set, ASR thread will pause listening (useful during enrollment)

_tts_executor = ThreadPoolExecutor(max_workers=1)  # serializes TTS calls
_tts_future = None

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

def _tts_worker(text):
    """Internal worker run inside executor (blocking). Uses audio_lock and tts_active."""
    global tts_engine
    # If engine not available, attempt init (best-effort)
    if tts_engine is None:
        try:
            init_tts()
        except Exception:
            pass

    if tts_engine is None:
        # fallback: print only
        print(f"[TTS-fallback] {text}")
        return

    with audio_lock:
        tts_active.set()
        try:
            # synchronous pyttsx3 call (safe because this runs in the executor thread)
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Error in tts worker: {e}")
        finally:
            tts_active.clear()
            # small sleep to ensure audio device drains before audio_lock is released to other users
            time.sleep(0.15)

def say(text, block=False):
    """
    Say text using serialized background TTS.
    - block=False (default) submits to single-worker executor and returns immediately.
    - block=True waits until the phrase finishes speaking.
    This keeps API backwards compatible while avoiding main-loop blocking.
    """
    global _tts_future
    if not text:
        return

    # simple logging
    print(f"[TTS] Enqueue: {text}")

    # Submit to executor
    future = _tts_executor.submit(_tts_worker, text)
    _tts_future = future

    if block:
        # wait, but allow KeyboardInterrupt to propagate by polling
        try:
            while True:
                try:
                    # tiny timeout so KeyboardInterrupt can interrupt
                    future.result(timeout=0.2)
                    break
                except Exception as e:
                    # Could be TimeoutError (not ready) or actual execution exception
                    # If it's a real exception from worker, re-raise it
                    if future.done() and future.exception() is not None:
                        raise future.exception()
                    # else keep waiting
                    continue
        except KeyboardInterrupt:
            # if user interrupts, best-effort: cancel future and clear flag
            with contextlib.suppress(Exception):
                future.cancel()
            tts_active.clear()
            raise

# helper to shutdown tts executor on program exit (call from main cleanup)
def shutdown_tts(wait_seconds=2.0):
    global _tts_executor, _tts_future
    try:
        _tts_executor.shutdown(wait=False)
        # allow up to wait_seconds for currently speaking phrase to finish
        if _tts_future is not None:
            try:
                _tts_future.result(timeout=wait_seconds)
            except Exception:
                pass
    except Exception:
        pass
# --------------------------
# VAD-enhanced recorder
# --------------------------
def record_wav_with_vad(device_index, duration, vad_mode=VAD_AGGRESSIVENESS):
    """
    Robust recording + WebRTC VAD wrapper.

    Returns:
        (fname_or_None, speech_ratio, used_samplerate)
    Notes:
        - Ensures audio passed to webrtcvad is one of {8000,16000,32000,48000}.
        - Uses scipy.signal.resample_poly when available; falls back to linear interpolation.
        - Writes wav only if speech_ratio > VAD_SPEECH_THRESHOLD.
    """
    SUPPORTED_VAD_RATES = (8000, 16000, 32000, 48000)
    FALLBACK_RATES = [16000, 48000, 32000, 8000, 44100, 22050]  # try preferred first
    try:
        dev = sd.query_devices(device_index)
    except Exception as e:
        print(f"[ASR] Failed to query device {device_index}: {e}")
        traceback.print_exc()
        return None, 0.0, None

    # Determine candidate samplerate to record at
    try:
        device_samplerate = int(dev.get('default_samplerate', 16000))
    except Exception:
        device_samplerate = 16000

    # Try to pick a record_rate that the device supports; prefer device default,
    # but make sure we can feed a supported rate to VAD (we will resample if needed).
    candidate_rates = [device_samplerate] + [r for r in FALLBACK_RATES if r != device_samplerate]

    # Validate input settings; pick first rate that check_input_settings accepts.
    record_rate = None
    for sr in candidate_rates:
        try:
            sd.check_input_settings(device=device_index, samplerate=sr, channels=1)
            record_rate = sr
            break
        except Exception:
            # not supported directly, we'll still allow recording at device_samplerate
            continue

    if record_rate is None:
        # Last resort: try device_samplerate without check (some devices accept but check fails)
        record_rate = device_samplerate
        print(f"[ASR] Warning: couldn't validate samplerate via check_input_settings; "
              f"attempting record at device default {record_rate} Hz")

    print(f"[ASR] Recording using device index {device_index} at {record_rate} Hz (device default: {device_samplerate} Hz)")

    channels = 1
    fname = None
    tmp_name = f"tmp_asr_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}.wav"

    try:
        # Acquire audio lock and record
        with audio_lock:
            rec = sd.rec(int(duration * record_rate), samplerate=record_rate, channels=channels,
                         dtype='int16', device=device_index)
            sd.wait()

        rec = np.asarray(rec)

        # If multichannel, collapse to mono by averaging channels (shouldn't happen with channels=1)
        if rec.ndim == 2 and rec.shape[1] > 1:
            rec = np.mean(rec, axis=1)

        # Flatten to 1-D int16 array
        rec = rec.flatten().astype(np.int16)

        # Quick RMS check for debugging
        try:
            rms = np.sqrt(np.mean((rec.astype(np.float32) / 32768.0) ** 2))
        except Exception:
            rms = 0.0
        print(f"[ASR DEBUG] Recorded {len(rec)} samples @ {record_rate} Hz, RMS={rms:.6f}")

        # Decide target sample rate for VAD (must be one of SUPPORTED_VAD_RATES)
        # Prefer 16000 if possible
        target_sr = 16000 if 16000 in SUPPORTED_VAD_RATES else SUPPORTED_VAD_RATES[0]
        if record_rate in SUPPORTED_VAD_RATES:
            target_sr = record_rate  # no resample required

        # If resampling needed, try best available resampler
        resampled = rec
        used_sr = record_rate
        if record_rate != target_sr:
            try:
                # Preferred: scipy.signal.resample_poly for quality
                from scipy.signal import resample_poly
                # convert to float32 for resampling
                rec_float = rec.astype(np.float32)
                # resample_poly expects shape (n,)
                gcd = np.gcd(target_sr, record_rate)
                up = target_sr // gcd
                down = record_rate // gcd
                res = resample_poly(rec_float, up, down)
                # scale/clip to int16
                res = np.clip(res, -32768, 32767).astype(np.int16)
                resampled = res
                used_sr = target_sr
                print(f"[ASR] Resampled from {record_rate} -> {target_sr} Hz using scipy.resample_poly")
            except Exception as e:
                # Fallback: simple linear interpolation resampler (lower quality)
                try:
                    print(f"[ASR] scipy resample not available or failed ({e}); using linear fallback")
                    import math
                    orig_len = len(rec)
                    new_len = int(math.ceil(len(rec) * target_sr / record_rate))
                    if orig_len < 2 or new_len < 1:
                        resampled = rec.copy()
                        used_sr = record_rate
                    else:
                        orig_idx = np.arange(orig_len)
                        new_idx = np.linspace(0, orig_len - 1, new_len)
                        res = np.interp(new_idx, orig_idx, rec.astype(np.float32))
                        resampled = np.clip(res, -32768, 32767).astype(np.int16)
                        used_sr = target_sr
                    print(f"[ASR] Linear resample from {record_rate} -> {target_sr} Hz (len {orig_len}->{len(resampled)})")
                except Exception as e2:
                    print(f"[ASR] Resampling fallback failed: {e2}. Proceeding with original rate {record_rate}")
                    resampled = rec
                    used_sr = record_rate

        # At this point `resampled` is int16 1-D and `used_sr` is the samplerate used for VAD.
        if used_sr not in SUPPORTED_VAD_RATES:
            # pick nearest supported rate and warn (but don't pass unsupported rate to VAD)
            # prefer 16000/48000
            fallback = 16000 if 16000 in SUPPORTED_VAD_RATES else SUPPORTED_VAD_RATES[0]
            print(f"[ASR] Warning: used_sr={used_sr} not supported by VAD. Trying fallback {fallback} Hz.")
            # try to resample to fallback quickly
            try:
                from scipy.signal import resample_poly
                gcd = np.gcd(fallback, used_sr)
                up = fallback // gcd
                down = used_sr // gcd
                res = resample_poly(resampled.astype(np.float32), up, down)
                resampled = np.clip(res, -32768, 32767).astype(np.int16)
                used_sr = fallback
            except Exception:
                # If scipy missing, last resort: set used_sr to record_rate and proceed (VAD may fail)
                used_sr = record_rate

        # Prepare frames for webrtcvad: must be 10/20/30 ms chunks
        frame_ms = 30
        frame_size = int(used_sr * frame_ms / 1000)
        if frame_size <= 0:
            print(f"[ASR] Invalid frame size {frame_size} for used_sr={used_sr}; aborting VAD")
            return None, 0.0, used_sr

        # Pad end so that length is multiple of frame_size
        n = len(resampled)
        remainder = n % frame_size
        if remainder != 0:
            pad_len = frame_size - remainder
            resampled = np.concatenate([resampled, np.zeros(pad_len, dtype=np.int16)])
            print(f"[ASR] Padded {pad_len} samples for full frames (new_len={len(resampled)})")

        vad = webrtcvad.Vad(vad_mode)
        voiced_frames = 0
        total_frames = 0

        for i in range(0, len(resampled), frame_size):
            frame = resampled[i:i + frame_size]
            if frame.size != frame_size:
                # pad if somehow still not full
                frame = np.pad(frame, (0, max(0, frame_size - frame.size)), 'constant')
            try:
                is_speech = vad.is_speech(frame.tobytes(), used_sr)
                if is_speech:
                    voiced_frames += 1
                total_frames += 1
            except Exception as e:
                # If webrtcvad fails (e.g. unsupported rate), bail out gracefully
                print(f"[ASR] VAD frame error: {e}; used_sr={used_sr}")
                total_frames = 0
                voiced_frames = 0
                break

        speech_ratio = (voiced_frames / total_frames) if total_frames > 0 else 0.0
        print(f"[ASR] VAD: voiced_frames={voiced_frames}, total_frames={total_frames}, ratio={speech_ratio:.2f}")

        # Save wav only if speech ratio exceeds threshold
        if speech_ratio > VAD_SPEECH_THRESHOLD:
            # Write file as int16 PCM with used_sr
            sf.write(tmp_name, resampled, used_sr, subtype='PCM_16')
            cleanup_files.append(tmp_name)
            print(f"[ASR] Saved speech to {tmp_name}")
            return tmp_name, speech_ratio, used_sr
        else:
            print(f"[ASR] Speech ratio below threshold ({VAD_SPEECH_THRESHOLD}); not saving.")
            return None, speech_ratio, used_sr

    except Exception as e:
        print(f"[ASR] Recording failed: {e}")
        traceback.print_exc()
        # Ensure tmp_name not left in cleanup_files
        try:
            if tmp_name in cleanup_files:
                cleanup_files.remove(tmp_name)
        except:
            pass
        return None, 0.0, None

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
        txt = r.recognize_google(aud, language="en-IN")
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
# ---------- Face detection + embedding helpers (replace existing versions) ----------
# Requires: mediapipe, deepface, numpy, cv2
import traceback
import threading

# persistent MediaPipe detector (create once)
_mp_face_detection_module = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils  # keep existing name

# create a single detector instance to reuse (avoid re-creating every frame)
# tune model_selection/min_detection_confidence if needed
try:
    _mp_face_detector = _mp_face_detection_module.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
except Exception as e:
    print(f"[MEDIAPIPE] Failed to initialize FaceDetection: {e}")
    _mp_face_detector = None

# DeepFace model cache (lazy-loaded)
_deepface_model_cache = {
    "model": None,
    "name": None,
    "lock": threading.Lock()
}

def detect_faces_mediapipe(frame, confidence=0.5):
    """
    Detect faces using a persistent MediaPipe FaceDetection instance.

    Returns:
        list of (x, y, width, height) tuples in pixel coordinates (same as old function).
    """
    faces = []
    try:
        if frame is None:
            return faces

        global _mp_face_detector
        if _mp_face_detector is None:
            # try to initialize lazily if previous attempt failed
            try:
                _mp_face_detector = _mp_face_detection_module.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=confidence
                )
            except Exception as e:
                print(f"[MEDIAPIPE] Could not initialize detector: {e}")
                return faces

        # MediaPipe expects RGB input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _mp_face_detector.process(rgb)

        if not results or not results.detections:
            return faces

        h, w = frame.shape[:2]
        for det in results.detections:
            # relative bounding box -> absolute pixels
            try:
                rbox = det.location_data.relative_bounding_box
                x = int(rbox.xmin * w)
                y = int(rbox.ymin * h)
                width = int(rbox.width * w)
                height = int(rbox.height * h)

                # clamp to frame
                x = max(0, x)
                y = max(0, y)
                width = max(0, min(width, w - x))
                height = max(0, min(height, h - y))

                # optional: skip very small boxes
                if width == 0 or height == 0:
                    continue

                faces.append((x, y, width, height))
            except Exception as e:
                print(f"[MEDIAPIPE] Detection parse error: {e}")
                continue

    except Exception as e:
        print(f"[MEDIAPIPE] Detection error: {e}")
        traceback.print_exc()

    return faces


def _ensure_deepface_model(model_name="ArcFace"):
    """
    Ensure a DeepFace model is loaded and cached. Returns model object or None.
    Thread-safe via lock to avoid races.
    """
    try:
        with _deepface_model_cache["lock"]:
            if _deepface_model_cache["model"] is not None and _deepface_model_cache["name"] == model_name:
                return _deepface_model_cache["model"]

            print(f"[DEEPFACE] Loading model '{model_name}' (this may take a moment)...")
            model = DeepFace.build_model(model_name)
            _deepface_model_cache["model"] = model
            _deepface_model_cache["name"] = model_name
            print("[DEEPFACE] Model loaded.")
            return model
    except Exception as e:
        print(f"[DEEPFACE] Failed to load model {model_name}: {e}")
        traceback.print_exc()
        return None


def _l2_normalize(emb):
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm <= 1e-10:
        return emb
    return emb / norm


def get_embedding_from_image(img_bgr, model_name="ArcFace", retry_count=2):
    """
    Robust wrapper around DeepFace.represent that tries multiple call signatures
    to handle mismatched DeepFace versions.

    Returns:
        np.ndarray embedding (float32) on success, or None on failure.
    """
    import inspect

    for attempt in range(retry_count):
        try:
            # Try the most common (older) signature first
            try:
                rep = DeepFace.represent(
                    img_path=img_bgr,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend='skip'
                )
            except TypeError as e1:
                # Try alternative signatures based on different DeepFace versions
                # 1) 'model' keyword
                try:
                    rep = DeepFace.represent(
                        img_path=img_bgr,
                        model=model_name,
                        enforce_detection=False,
                        detector_backend='skip'
                    )
                except TypeError as e2:
                    # 2) positional args: DeepFace.represent(img, model_name, ...)
                    try:
                        rep = DeepFace.represent(img_bgr, model_name, False, 'skip')
                    except TypeError as e3:
                        # 3) finally try the simplest positional call DeepFace.represent(img)
                        #    sometimes represent() determines defaults internally.
                        try:
                            rep = DeepFace.represent(img_bgr)
                        except Exception as e4:
                            # If we reach here, none of the calling conventions worked.
                            # Raise original TypeError to outer except and allow retry.
                            raise e1

            # DeepFace.represent may return a list of dicts or a single dict
            if isinstance(rep, list) and len(rep) > 0:
                rep = rep[0]

            # rep should be a dict with "embedding" key
            if isinstance(rep, dict) and "embedding" in rep:
                emb = np.asarray(rep["embedding"], dtype=np.float32)
                return emb
            else:
                # Unexpected return format
                print(f"[DEEPFACE] Unexpected represent() return type: {type(rep)}. Content keys: {getattr(rep, 'keys', lambda: None)()}")
                return None

        except Exception as e:
            # On failure, either retry (up to retry_count) or return None
            if attempt < retry_count - 1:
                print(f"[DEEPFACE] Retry {attempt + 1}/{retry_count} after error: {e}")
                time.sleep(0.1)
                continue
            else:
                print(f"[DEEPFACE] Embedding failed after {retry_count} attempts: {e}")
                traceback.print_exc()
                return None

    return None


# helper to gracefully close the MediaPipe detector on shutdown
def close_mediapipe_detector():
    global _mp_face_detector
    try:
        if _mp_face_detector is not None:
            _mp_face_detector.close()
            _mp_face_detector = None
            print("[MEDIAPIPE] Detector closed.")
    except Exception as e:
        print(f"[MEDIAPIPE] Error closing detector: {e}")

# End of replacement block

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
    """
    ASR listener thread with robust handling:
    - Respects tts_active and asr_pause (so it won't listen while TTS or enrollment runs).
    - Handles different return shapes from record_wav_with_vad (2-tuple or 3-tuple).
    - Cleans up temp files safely (only if they exist).
    - Puts final transcribed text into the queue (if any).
    """
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
                # Respect TTS and explicit pause (enrollment)
                if tts_active.is_set() or asr_pause.is_set():
                    time.sleep(0.1)
                    continue

                # Call the recorder. record_wav_with_vad may return:
                #  - (fname, speech_ratio, used_sr)
                #  - (fname, speech_ratio)  (older callers)
                #  - None / (None, 0.0, None)
                try:
                    result = record_wav_with_vad(INPUT_DEVICE_INDEX, ASR_RECORD_SECONDS)
                except Exception as e:
                    print(f"[ASR] Recording failed (exception): {e}")
                    traceback.print_exc()
                    time.sleep(0.1)
                    continue

                # Normalize return to (fname, speech_ratio)
                fname = None
                speech_ratio = 0.0
                if result is None:
                    fname = None
                elif isinstance(result, tuple):
                    if len(result) >= 2:
                        fname = result[0]
                        speech_ratio = result[1]
                    elif len(result) == 1:
                        fname = result[0]
                        speech_ratio = 0.0
                    else:
                        fname = None
                else:
                    # Unexpected return type
                    print("[ASR] Unexpected recorder return type; skipping")
                    time.sleep(0.05)
                    continue

                if not fname:
                    consecutive_silence += 1
                    if consecutive_silence % 10 == 0:
                        print(f"[ASR] Listening... (silence count: {consecutive_silence})")
                    time.sleep(0.05)
                    continue

                consecutive_silence = 0

                # Transcribe safely
                txt = ""
                try:
                    txt = transcribe_wav_file(fname)
                except Exception as e:
                    print(f"[ASR] Transcription exception: {e}")
                    traceback.print_exc()
                    txt = ""

                # Remove temp file if it exists and is in our cleanup list
                try:
                    if fname in cleanup_files:
                        try:
                            cleanup_files.remove(fname)
                        except ValueError:
                            pass
                    if os.path.exists(fname):
                        os.remove(fname)
                except Exception as e:
                    print(f"[ASR] Warning: failed to remove temp file {fname}: {e}")

                if txt:
                    print(f"[ASR] >>> HEARD: \"{txt}\"")
                    try:
                        self.q.put(txt)
                    except Exception as e:
                        print(f"[ASR] Warning: failed to enqueue command: {e}")
                else:
                    print(f"[ASR] (Speech detected but could not transcribe)")

                time.sleep(0.05)
            except Exception as e:
                # Keep thread alive on unexpected errors
                print(f"[ASR] Error in listener loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)

    def stop(self):
        self.running = False
# --------------------------
# Enrollment process (FIXED)
# --------------------------
def perform_enrollment(cam, enrolled_db):
    """
    Robust enrollment flow (typed name + multi-pose captures).

    Behavior / safety:
    - Pauses ASR early by setting `asr_pause` so listener won't start new recordings.
      (Note: ASR may finish an already-started record; we wait briefly to reduce races.)
    - Avoids speaking per-pose prompts (no repeated TTS during capture) to prevent
      TTS being re-captured by the mic & VAD. Uses on-screen text for pose guidance.
    - Requires at least 2 valid captures; validates crop size and contiguity.
    - Uses get_embedding_from_image() to compute embeddings (must exist and be robust).
    - Saves enrollment under a sanitized lowercase key via save_enrollment().
    - Always clears `asr_pause` in a finally block so ASR resumes.
    - Returns the updated enrolled_db on success (or original on cancel/failure).
    """
    # Defensive: ensure globals exist
    global asr_pause, tts_active

    # Pause ASR as early as possible to avoid extra recordings.
    try:
        asr_pause.set()
    except Exception:
        # If asr_pause isn't available, continue but warn
        print("[ENROLL] Warning: asr_pause Event not available; proceeding without ASR pause")

    # Give ASR loop a short moment to observe the pause and finish any in-flight record.
    time.sleep(0.35)

    try:
        # Short audible/printed instruction once (safe): announce then request typed name.
        say("Starting enrollment. Please type the person's name in the terminal.")
        time.sleep(0.15)

        print("\n[ENROLL] Please type the name for enrollment (or leave blank to cancel):")
        try:
            spoken_name = input("Name to enroll: ").strip()
        except Exception as e:
            print(f"[ENROLL] Input error: {e}")
            say("Could not read name. Enrollment cancelled.")
            return enrolled_db

        if not spoken_name:
            say("Enrollment cancelled.")
            print("[ENROLL] Cancelled by user (no name provided).")
            return enrolled_db

        # sanitize and lower-case storage name (file-system safe)
        spoken_name_key = "".join(ch for ch in spoken_name if ch.isalnum() or ch in (" ", "_", "-")).strip().lower()
        if not spoken_name_key:
            say("Provided name is invalid. Enrollment cancelled.")
            print("[ENROLL] Provided name invalid after sanitization.")
            return enrolled_db

        print(f"[ENROLL] Proceeding to enroll: '{spoken_name_key}'")
        # Give a single verbal notification (not per-pose)
        say(f"Enrolling {spoken_name}. Follow the instructions on the screen. Do not speak while I capture photos.")

        # Pose prompts displayed on-screen only
        pose_prompts = [
            "Look straight at the camera",
            "Turn your head slightly to the left",
            "Turn your head slightly to the right",
            "Tilt your head slightly up",
            "Tilt your head slightly down"
        ]

        captures = []
        say("Get ready. Position your face inside the green box.")
        time.sleep(0.6)

        # Capture loop (runs in main thread; uses provided cam)
        for idx, prompt in enumerate(pose_prompts[:ENROLL_PHOTOS]):
            best_crop = None
            best_area = 0

            # Display initial guidance for this pose (no TTS)
            start_time = time.time()
            while time.time() - start_time < 1.5:
                ret, frame = cam.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                faces = detect_faces_mediapipe(frame, confidence=0.65)

                # UI guidance (on-screen)
                display = frame.copy()
                h, w = display.shape[:2]
                cv2.putText(display, f"Enrollment: {idx + 1}/{ENROLL_PHOTOS}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, prompt, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Draw green target box to encourage consistent framing
                cx1, cy1 = int(w * 0.3), int(h * 0.2)
                cx2, cy2 = int(w * 0.7), int(h * 0.8)
                cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)

                # Draw detected faces and pick the largest valid one
                for (x, y, fw, fh) in faces:
                    # Basic coordinate safety
                    if fw <= 0 or fh <= 0:
                        continue
                    x0 = max(0, x); y0 = max(0, y)
                    x1 = min(w, x + fw); y1 = min(h, y + fh)
                    crop_h = y1 - y0; crop_w = x1 - x0
                    if crop_h <= 0 or crop_w <= 0:
                        continue

                    area = crop_h * crop_w
                    cv2.rectangle(display, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    if area > best_area:
                        # minimal resolution threshold for crop
                        if crop_h >= 40 and crop_w >= 40:
                            candidate = frame[y0:y1, x0:x1].copy()
                            # ensure contiguous uint8 array
                            candidate = np.ascontiguousarray(candidate.astype(np.uint8))
                            if candidate.size > 0:
                                best_area = area
                                best_crop = candidate

                cv2.imshow("Enrollment (ESC to cancel)", display)
                k = cv2.waitKey(30) & 0xFF
                if k == 27:
                    say("Enrollment cancelled.")
                    cv2.destroyAllWindows()
                    print("[ENROLL] Cancelled by user (ESC pressed).")
                    return enrolled_db

            # End capture attempt for this pose
            if best_crop is not None and best_crop.size > 0:
                # Additional safety: require a reasonable resolution
                if best_crop.shape[0] < 40 or best_crop.shape[1] < 40:
                    print(f"[ENROLL] Skipped capture (too small) shape={best_crop.shape}")
                else:
                    captures.append(best_crop)
                    # Use a non-voice confirmation to avoid microphone pickup; small pause instead
                    print(f"[ENROLL] Captured pose {idx + 1}, area={best_area}, shape={best_crop.shape}")
                    time.sleep(0.25)
            else:
                print(f"[ENROLL] No clear face detected for pose {idx + 1}; skipping.")
                time.sleep(0.15)

        # Close the enrollment window(s)
        cv2.destroyAllWindows()

        # Need at least two good captures for embedding consistency check
        if len(captures) < 2:
            say("Enrollment failed. Not enough valid captures. Please try again.")
            print("[ENROLL] Failed: not enough valid captures.")
            return enrolled_db

        # Compute embeddings (single TTS to announce the step)
        say("Computing face embeddings. Please wait.")
        embeddings = []
        for i, img in enumerate(captures):
            print(f"[ENROLL] Computing embedding {i + 1}/{len(captures)}...")
            try:
                emb = get_embedding_from_image(img)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    print(f"[ENROLL] get_embedding_from_image returned None for capture {i + 1}")
            except Exception as e:
                print(f"[ENROLL] Exception computing embedding for capture {i + 1}: {e}")
                traceback.print_exc()

        if len(embeddings) < 2:
            say("Enrollment failed. Could not extract reliable face features. Please try again.")
            print("[ENROLL] Failed: insufficient embeddings extracted.")
            return enrolled_db

        # Consistency check (intra-person distances)
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                distances.append(cosine_distance(embeddings[i], embeddings[j]))
        avg_dist = float(np.mean(distances)) if distances else 1.0
        print(f"[ENROLL] Average intra-person distance: {avg_dist:.4f}")
        if avg_dist > 0.5:
            say("Warning: face captures show high variation. Enrollment may be less reliable.")

        # Save enrollment (use sanitized key)
        saved = False
        try:
            saved = save_enrollment(spoken_name_key, embeddings)
        except Exception as e:
            print(f"[ENROLL] Error saving enrollment: {e}")
            traceback.print_exc()
            saved = False

        if saved:
            # reload DB to update in-memory state
            enrolled_db = load_enrolled_db()
            say(f"Enrollment complete for {spoken_name}. {len(embeddings)} embeddings saved.")
            print(f"[ENROLL] Success: '{spoken_name_key}' enrolled with {len(embeddings)} embeddings")
        else:
            say("Failed to save enrollment data. Please check disk permissions and try again.")
            print("[ENROLL] Failed to save enrollment data.")

        return enrolled_db

    finally:
        # Always resume ASR regardless of outcome
        try:
            asr_pause.clear()
        except Exception:
            pass
        # Small delay to avoid immediate ASR records while cleaning up
        time.sleep(0.15)
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
        print("[WARN] TTS initialization failed; proceeding in degraded mode (no TTS).")

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

    # Create persistent MediaPipe detector (reuse for all frames)
    face_detector = None
    try:
        face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    except Exception as e:
        print(f"[INIT] Failed to create MediaPipe FaceDetection: {e}")
        face_detector = None

    # State variables
    armed = False
    state = "IDLE"
    unrec_start = None           # first time unrecognized presence observed
    last_message_time = 0        # time when we last spoken (used for cooldown)
    last_face_process = 0.0

    # limits
    MAX_CMD_PER_LOOP = 3

    say("AI Guard Agent ready. Say 'guard mode on' to activate monitoring.")
    print("\n[INFO] Press ESC in camera window to exit.")
    print("[INFO] Say 'enroll' to add a trusted person (only when disarmed).")

    try:
        while True:
            # Process up to a few commands per loop to avoid starving frame processing
            cmds_processed = 0
            while not cmd_q.empty() and cmds_processed < MAX_CMD_PER_LOOP:
                try:
                    txt = cmd_q.get_nowait()
                except queue.Empty:
                    break
                cmds_processed += 1
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
                        last_message_time = 0
                        say("Guard mode deactivated.")
                        print("[STATE] DISARMED")
                    else:
                        print("[CMD] Already disarmed")
                    continue

                # Activation - fuzzy matching
                is_activation, matched_phrase = matches_activation_command(txt, activation_phrases)
                if is_activation:
                    if not armed:
                        armed = True
                        state = "ARMED"
                        unrec_start = None
                        last_message_time = 0
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
                # sometimes camera fails transiently; wait a bit and continue
                time.sleep(0.05)
                continue

            # Display frame with status
            display = frame.copy()
            h, w = display.shape[:2]

            if armed:
                cv2.putText(display, "GUARD MODE: ARMED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(display, "GUARD MODE: DISARMED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2)

            # Face detection and recognition (only when armed)
            current_time = time.time()
            if armed and (current_time - last_face_process) > FACE_PROCESS_INTERVAL:
                last_face_process = current_time

                faces = []
                try:
                    # Prefer using the persistent detector (faster than constructing each time)
                    if face_detector is not None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detector.process(rgb)
                        if results and results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                # clamp
                                x = max(0, x)
                                y = max(0, y)
                                width = min(width, w - x)
                                height = min(height, h - y)
                                if width > 0 and height > 0:
                                    faces.append((x, y, width, height))
                    else:
                        # fallback to function if detector not available
                        faces = detect_faces_mediapipe(frame, confidence=0.5)
                except Exception as e:
                    print(f"[CAM] Face detection error: {e}")
                    faces = []

                if len(faces) == 0:
                    # No faces detected - reset unrecognized tracking
                    if unrec_start is not None:
                        print("[GUARD] No faces detected - resetting unrecognized timer")
                    unrec_start = None
                    state = "ARMED"
                else:
                    any_unrecognized = False

                    for (x, y, fw, fh) in faces:
                        # defensive cropping (avoid slicing outside bounds)
                        x2 = min(x + fw, w)
                        y2 = min(y + fh, h)
                        x1 = max(0, x)
                        y1 = max(0, y)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        face_crop = frame[y1:y2, x1:x2].copy()
                        if face_crop.size == 0:
                            continue

                        emb = None
                        try:
                            emb = get_embedding_from_image(face_crop)
                        except Exception as e:
                            print(f"[GUARD] Embedding extraction error: {e}")
                            emb = None

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

                        # Draw bounding box and label
                        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, f"{label} ({dist:.2f})", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        if label == "unknown":
                            any_unrecognized = True

                    # Collective escalation logic (fixed timing)
                    if any_unrecognized:
                        if unrec_start is None:
                            unrec_start = current_time
                            state = "UNRECOGNIZED_DETECTED"
                            # immediate greeting
                            say("Hello. Please identify yourself.")
                            print("[ESCALATE] Initial greeting")
                            last_message_time = current_time
                        else:
                            elapsed = current_time - unrec_start
                            time_since_last_msg = current_time - last_message_time

                            # WARNING phase: use elapsed only (no cooldown blocking the very first warning)
                            if state == "UNRECOGNIZED_DETECTED" and elapsed > WARNING_AFTER:
                                state = "WARNING"
                                say("You are not authorized. Please leave immediately.")
                                print("[ESCALATE] Warning issued")
                                last_message_time = current_time

                            # ESCALATION phase
                            if state == "WARNING" and elapsed > ESCALATE_AFTER:
                                state = "ESCALATED"
                                say("This is your final warning. This area is monitored. Leave now.")
                                print("[ESCALATE] Final warning")
                                last_message_time = current_time

                            # Repeated escalations (use cooldown)
                            if state == "ESCALATED" and time_since_last_msg > ESCALATION_COOLDOWN:
                                say("Unauthorized presence detected. Authorities will be notified.")
                                print("[ESCALATE] Repeated warning")
                                last_message_time = current_time
                    else:
                        # All faces recognized - reset
                        if unrec_start is not None:
                            print("[GUARD] All faces recognized - resetting")
                            unrec_start = None
                            state = "ARMED"

            # Display state info
            if state != "IDLE" and state != "ARMED":
                cv2.putText(display, f"State: {state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("AI Guard - Press ESC to exit", display)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                # stop main loop; say goodbye (non-blocking), then break
                say("Shutting down. Goodbye.")
                break

            # tiny sleep to yield
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
        traceback.print_exc()
    finally:
        print("[MAIN] Cleaning up...")
        # Stop ASR first to avoid new commands while shutting down
        stop_event.set()
        asr.stop()
        asr.join(timeout=3.0)

        # release camera and windows
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

        # shutdown face detector if created
        try:
            if face_detector is not None:
                face_detector.close()
        except Exception:
            pass

        # shutdown tts executor (allow short time for last phrase)
        shutdown_tts(wait_seconds=1.0)

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