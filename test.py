#!/usr/bin/env python3
"""
test_components.py - Test individual AI Guard components

Usage: python test_components.py
"""

import os
# Force Qt to use xcb (avoids Wayland plugin error on many Linux installs)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import sys
import time
import cv2
import sounddevice as sd
import soundfile as sf
import pyttsx3
import mediapipe as mp
import numpy as np
from datetime import datetime

# Resampling helper
try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None  # will check before use

# --- constants ---
INPUT_DEVICE_INDEX = 4  # device index you tested earlier
TARGET_VAD_SR = 16000

# ---------------------
# Helpers
# ---------------------
def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def record_with_device_default(device_index, duration_seconds, channels=1, want_sr=None):
    """
    Record from `device_index` using the device's default sample rate, then optionally
    resample to `want_sr`. Returns (tmp_wav_path, audio_array_int16, samplerate).
    On error returns (None, None, None).
    """
    try:
        dev = sd.query_devices(device_index)
    except Exception as e:
        print(f"[REC] query_devices failed for index {device_index}: {e}")
        return None, None, None

    sr_device = int(dev.get("default_samplerate", 48000))
    device_channels = int(dev.get("max_input_channels", 1))
    channels = min(max(1, channels), max(1, device_channels))

    fname = f"tmp_rec_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}.wav"
    try:
        # Record using device default sample rate to avoid PortAudio invalid sample rate errors
        rec = sd.rec(int(duration_seconds * sr_device), samplerate=sr_device, channels=channels, dtype="int16", device=device_index)
        sd.wait()
    except Exception as e:
        print(f"[REC] Failed to record from device {device_index} at {sr_device} Hz: {e}")
        return None, None, None

    rec = np.asarray(rec)
    if rec.ndim > 1:
        mono = rec.mean(axis=1).astype(np.int16)
    else:
        mono = rec.astype(np.int16)

    # Save the device-rate file (useful for debugging). Overwrite if exists.
    try:
        sf.write(fname, mono, sr_device)
    except Exception as e:
        print(f"[REC] Failed to write temporary WAV {fname}: {e}")

    # If the caller requests a specific sample rate, resample in memory
    if want_sr and want_sr != sr_device:
        if resample_poly is None:
            print("[REC] scipy not available: cannot resample. Returning device-rate audio.")
            return fname, mono, sr_device
        # polyphase resample
        gcd = np.gcd(sr_device, want_sr)
        up = want_sr // gcd
        down = sr_device // gcd
        try:
            mono_resampled = resample_poly(mono, up, down).astype(np.int16)
            return fname, mono_resampled, want_sr
        except Exception as e:
            print(f"[REC] Resampling failed: {e}. Returning device-rate audio.")
            return fname, mono, sr_device

    return fname, mono, sr_device


# ---------------------
# Tests
# ---------------------
def test_audio_devices():
    """Test 1: List and validate audio devices"""
    print_header("TEST 1: Audio Devices")
    try:
        devices = sd.query_devices()
        print("\nAvailable Audio Devices:")
        for i, dev in enumerate(devices):
            input_ch = dev.get("max_input_channels", 0)
            output_ch = dev.get("max_output_channels", 0)
            print(f"  [{i}] {dev['name']}")
            print(f"      Input channels: {input_ch}, Output channels: {output_ch}")
            print(f"      Sample rate: {dev.get('default_samplerate', 'N/A')} Hz")

        print(f"\n[INFO] Checking INPUT_DEVICE_INDEX = {INPUT_DEVICE_INDEX}...")
        try:
            dev4 = sd.query_devices(INPUT_DEVICE_INDEX)
            if dev4.get("max_input_channels", 0) > 0:
                print(f"✓ Device {INPUT_DEVICE_INDEX} is valid: {dev4['name']}")
                return True
            else:
                print(f"✗ Device {INPUT_DEVICE_INDEX} has no input channels: {dev4['name']}")
                return False
        except Exception as e:
            print(f"✗ Device {INPUT_DEVICE_INDEX} not found: {e}")
            return False
    except Exception as e:
        print(f"✗ Error listing audio devices: {e}")
        return False


def test_microphone():
    """Test 2: Record from microphone (uses device default SR + resample to 16k)"""
    print_header("TEST 2: Microphone Recording")
    try:
        print(f"\n[INFO] Recording ~2 seconds from device {INPUT_DEVICE_INDEX} (device-default SR, resampled to {TARGET_VAD_SR} Hz)...")
        print("[INFO] Please speak into the microphone...")

        fname, audio, sr = record_with_device_default(INPUT_DEVICE_INDEX, duration_seconds=2.0, want_sr=TARGET_VAD_SR)
        if audio is None:
            print("✗ Recording failed or no audio captured.")
            return False

        # compute RMS on float-normalized signal
        rms = np.sqrt(np.mean((audio.astype("float32") / 32768.0) ** 2))
        print(f"\n✓ Recording successful")
        print(f"  Sampling rate (used for ASR/VAD): {sr} Hz")
        print(f"  RMS level: {rms:.6f}")

        try:
            os.remove(fname)
        except:
            pass

        if rms > 0.001:
            print("  ✓ Audio signal detected (good)")
            return True
        else:
            print("  ✗ Very weak signal (check microphone)")
            return False
    except Exception as e:
        print(f"✗ Error during microphone test: {e}")
        return False


def test_tts():
    """Test 3: Text-to-Speech"""
    print_header("TEST 3: Text-to-Speech")
    try:
        print("\n[INFO] Initializing TTS engine...")
        engine = pyttsx3.init()

        print("[INFO] Speaking test message...")
        test_msg = "AI Guard system test. Can you hear me?"
        print(f"  Message: \"{test_msg}\"")

        engine.say(test_msg)
        engine.runAndWait()

        print("✓ TTS test complete")

        response = input("\n[QUESTION] Did you hear the message? (y/n): ").strip().lower()
        return response == "y"
    except Exception as e:
        print(f"✗ Error initializing or running TTS: {e}")
        return False


def test_camera():
    """Test 4: Camera access"""
    print_header("TEST 4: Camera Access")
    try:
        print("\n[INFO] Opening camera 0...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open camera")
            return False

        print("✓ Camera opened")
        print("[INFO] Capturing frame... Press any key to close window")

        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Frame captured: {w}x{h}")
            cv2.putText(
                frame,
                "Camera Test - Press any key",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            return True
        else:
            print("✗ Could not capture frame")
            cap.release()
            return False
    except Exception as e:
        print(f"✗ Error during camera test: {e}")
        return False


def test_mediapipe():
    """Test 5: MediaPipe face detection"""
    print_header("TEST 5: MediaPipe Face Detection")
    try:
        print("\n[INFO] Initializing MediaPipe...")
        mp_face_detection = mp.solutions.face_detection

        print("[INFO] Opening camera for face detection...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open camera")
            return False

        print("✓ Camera opened")
        print("[INFO] Show your face to the camera. Press ESC to exit.")

        face_detected = False
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            start_time = time.time()
            while time.time() - start_time < 10:  # 10 second test
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                display = frame.copy()

                if results.detections:
                    face_detected = True
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)

                        cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        cv2.putText(display, "Face Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(display, "MediaPipe Test - Press ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("MediaPipe Face Detection Test", display)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

        if face_detected:
            print("✓ Face detection working")
            return True
        else:
            print("✗ No face detected (try positioning your face better)")
            return False
    except Exception as e:
        print(f"✗ Error during MediaPipe test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deepface():
    """Test 6: DeepFace embedding extraction"""
    print_header("TEST 6: DeepFace Embedding")
    try:
        print("\n[INFO] Testing DeepFace...")
        print("[INFO] This will take a moment on first run (downloading models)...")

        from deepface import DeepFace

        print("[INFO] Opening camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("✗ Could not open camera")
            return False

        print("[INFO] Capturing face... Look at the camera")
        time.sleep(1)

        # Capture multiple frames to get a good one
        best_frame = None
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                best_frame = frame
                time.sleep(0.1)

        cap.release()

        if best_frame is None:
            print("✗ Could not capture frame")
            return False

        print("[INFO] Extracting face embedding...")
        try:
            rep = DeepFace.represent(img_path=best_frame, model_name="ArcFace", enforce_detection=False, detector_backend="skip")

            if isinstance(rep, list):
                rep = rep[0]

            embedding = np.array(rep["embedding"])
            print(f"✓ Embedding extracted successfully")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  Sample values: {embedding[:5]}")
            return True
        except Exception as e:
            print(f"✗ DeepFace error: {e}")
            return False
    except Exception as e:
        print(f"✗ Error initializing DeepFace test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_webrtcvad():
    """Test 7: WebRTC VAD (uses device default SR + resample)"""
    print_header("TEST 7: WebRTC VAD")
    try:
        import webrtcvad

        print("\n[INFO] Initializing WebRTC VAD...")
        vad = webrtcvad.Vad(3)

        print("[INFO] Recording ~2 seconds (device default SR then resample to 16k)...")
        fname, audio, sr = record_with_device_default(INPUT_DEVICE_INDEX, duration_seconds=2.0, want_sr=TARGET_VAD_SR)
        if audio is None:
            print("✗ Recording failed or silent.")
            return False

        frame_duration = 30  # ms
        frame_size = int(TARGET_VAD_SR * frame_duration / 1000)

        voiced_frames = 0
        total_frames = 0

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i : i + frame_size].tobytes()
            total_frames += 1
            try:
                if vad.is_speech(frame, TARGET_VAD_SR):
                    voiced_frames += 1
            except Exception:
                pass

        speech_ratio = voiced_frames / total_frames if total_frames > 0 else 0

        print(f"\n✓ VAD analysis complete")
        print(f"  Total frames: {total_frames}")
        print(f"  Voiced frames: {voiced_frames}")
        print(f"  Speech ratio: {speech_ratio:.2%}")

        try:
            os.remove(fname)
        except:
            pass

        if speech_ratio > 0.1:
            print("  ✓ Speech detected")
            return True
        else:
            print("  ✗ No significant speech detected")
            return False
    except Exception as e:
        print(f"✗ Error during VAD test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  AI GUARD AGENT - COMPONENT TESTING")
    print("=" * 60)
    print("\nThis will test all system components.")
    print("Make sure your microphone and camera are connected.")

    input("\nPress ENTER to start tests...")

    results = {}

    # Run tests
    results["Audio Devices"] = test_audio_devices()
    results["Microphone"] = test_microphone()
    results["TTS"] = test_tts()
    results["Camera"] = test_camera()
    results["MediaPipe"] = test_mediapipe()
    results["DeepFace"] = test_deepface()
    results["WebRTC VAD"] = test_webrtcvad()

    # Summary
    print_header("TEST SUMMARY")
    print()
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:<20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED - System ready!")
    else:
        print("  ✗ SOME TESTS FAILED - Check errors above")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())