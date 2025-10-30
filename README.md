## Assignment #2 EE782 - AI Guard Agent

## Team
- Dhruv Meena 22b1279
- Madhava Sriram 22b1233

## To run

After cloning repo and making it current working directory.

### Activate virtual environment
```bash
python -m venv venv
venv/scripts/activate
```

### Install dependencies
```bash
pip install -requirements.txt
```
Note: You will need Pyaudio and webcrtvad modules. Install with conda/miniconda.

### Run Unit Tests
```bash
python test.py
```

### Run Guard Agent
```bash
python main.py
```

## Video
[Link](https://drive.google.com/drive/folders/1-zq1_mV51Z7O2tidyDMIm1MFsllde5FF?usp=sharing)

## FlowChart

```mermaid
flowchart TD
  %% Main program and initialization
  Start["Start / main()"]
  Init["Initialize subsystems\n(setup_audio, init_tts, load DB)"]
  CameraOpen["Open Camera (cv2.VideoCapture)"]
  ASRStart["Start ASRListener thread"]
  MainLoop["guard_main_loop() (camera + state machine)"]
  Cleanup["Cleanup & Exit"]

  Start --> Init --> CameraOpen --> ASRStart --> MainLoop --> Cleanup

  %% ASR subsystem
  subgraph ASR [ASR Thread]
    direction TB
    ASRLoop["Loop: wait (tts_active/asr_pause)"]
    Record["record_wav_with_vad()\n(record -> resample -> VAD)"]
    SavedFile["Save WAV if speech_ratio > thresh"]
    Transcribe["transcribe_wav_file() (SpeechRecognition)"]
    PutCmd["Put text into command queue"]
    ASRLoop --> Record --> SavedFile --> Transcribe --> PutCmd
  end
  ASRStart --> ASR

  %% TTS
  subgraph TTS [TTS / Audio Output]
    direction TB
    TTS_Engine["pyttsx3 engine\nthread-safe via audio_lock"]
    TTS_Say["say(text) -> acquires audio_lock"]
  end
  MainLoop --- TTS_Say
  PutCmd --> MainLoop

  %% Enrollment flow
  subgraph ENROLL [Enrollment]
    direction TB
    EnrollCmd["Trigger: matches_command('enroll')"]
    PauseASR["Set asr_pause / tts_active"]
    GetName["Ask user to type name (terminal input)"]
    CapturePoses["Capture ENROLL_PHOTOS using camera\n(detect_faces_mediapipe)"]
    ValidateCrops["Validate size & contiguity"]
    Embeddings["get_embedding_from_image() x N\n(DeepFace)"]
    SaveEnroll["save_enrollment(name, embeddings)"]
    ResumeASR["Clear asr_pause"]
  end

  EnrollCmd --> PauseASR --> GetName --> CapturePoses --> ValidateCrops --> Embeddings --> SaveEnroll --> ResumeASR
  MainLoop -->|enroll command| EnrollCmd

  %% Vision / Recognition logic when armed
  subgraph VISION [Vision / Recognition]
    direction TB
    DetectFaces["detect_faces_mediapipe(frame)"]
    GetEmb["get_embedding_from_image(face_crop)"]
    Match["match_embedding(enrolled_db, emb)"]
    Label["Label (name or unknown)\ndraw bbox"]
    EscalateLogic["Escalation logic:\nUNRECOGNIZED -> WARNING -> ESCALATE"]
  end

  MainLoop -->|if armed| DetectFaces --> GetEmb --> Match --> Label
  Label --> EscalateLogic
  EscalateLogic --> TTS_Say

  %% Storage and DB
  subgraph DB [Enrollment Storage]
    direction TB
    ENROLL_DIR["ENROLL_DIR (npz files)"]
    LoadDB["load_enrolled_db()"]
    SaveFile["np.savez_compressed(name.npz)"]
  end

  SaveEnroll --> SaveFile --> ENROLL_DIR
  Init --> LoadDB
  LoadDB --> MainLoop

  %% Signals and cleanup
  Cleanup -->|on exit| StopASR["stop_event.set(); asr.stop()"]
  StopASR --> JoinASR["asr.join(); cap.release(); cv2.destroyAllWindows()"]
  JoinASR --> End["End"]
```
