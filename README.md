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

flowchart TD

%% ========= SYSTEM INITIALIZATION =========
Start([Start / main()])
Init[Initialize logging & config]
Audio[Setup audio devices]
TTSInit[Init TTS engine]
LoadPhrases[Load activation phrases]
LoadDB[Load enrolled face DB]
OpenCam[Open camera (cv2.VideoCapture)]
Threads[Start ASRListener thread, events & locks]
MainLoop[Enter guard_main_loop()]

Start --> Init
Init --> Audio
Audio --> TTSInit
TTSInit --> LoadPhrases
LoadPhrases --> LoadDB
LoadDB --> OpenCam
OpenCam --> Threads
Threads --> MainLoop

%% ========= ASR THREAD =========
subgraph ASRListener["ASR Listener Thread"]
  A1[Loop: while not stopped]
  A2{tts_active or asr_pause?}
  A3[Wait 0.1s]
  A4[record_wav_with_vad()]
  A5{File saved?}
  A6[Increment silence counter]
  A7[transcribe_wav_file()]
  A8{Text recognized?}
  A9[Put text → cmd_queue]

  A1 --> A2
  A2 -->|Yes| A3
  A2 -->|No| A4
  A4 --> A5
  A5 -->|No| A6
  A5 -->|Yes| A7
  A7 --> A8
  A8 -->|Yes| A9
  A8 -->|No| A6
end

%% ========= MAIN LOOP =========
subgraph MAIN["Main Guard Loop"]
  M1[Poll cmd_queue]
  M2{Command type?}
  Activate[Arm system; say("Guard mode activated")]
  Deactivate[Disarm system; say("Guard mode deactivated")]
  EnrollCmd[Perform enrollment]
  IgnoreCmd[Log / ignore unknown commands]
  Frame[Capture camera frame]
  Process[If armed → detect_faces_mediapipe()]
  Faces{Faces found?}
  ForEach[For each face → get_embedding → match_embedding()]
  Match{Matched below threshold?}
  Known[Recognized; draw green box]
  Unknown[Unknown; draw red box; any_unrecognized=True]
  Escalate[Run escalation logic]
  Warn[say("Identify yourself / leave now")]
  EndLoop[Loop back to next frame]

  M1 --> M2
  M2 -->|Activation| Activate
  M2 -->|Deactivation| Deactivate
  M2 -->|Enroll| EnrollCmd
  M2 -->|Other| IgnoreCmd

  Frame --> Process
  Process --> Faces
  Faces -->|No| EndLoop
  Faces -->|Yes| ForEach
  ForEach --> Match
  Match -->|Yes| Known
  Match -->|No| Unknown
  Unknown --> Escalate
  Escalate --> Warn
  Warn --> EndLoop
end

%% ========= ENROLLMENT FLOW =========
subgraph ENROLLMENT["Enrollment Process"]
  E1[Pause ASR + TTS]
  E2[Prompt user for name (input)]
  E3[Capture multiple poses via camera]
  E4[Compute embeddings (DeepFace)]
  E5{≥2 valid embeddings?}
  E6[save_enrollment(name, embeddings)]
  E7[Reload enrolled DB]
  E8[Resume ASR]

  E1 --> E2 --> E3 --> E4 --> E5
  E5 -->|No| E8
  E5 -->|Yes| E6 --> E7 --> E8
end

%% ========= CLEANUP =========
Exit{Exit condition (ESC or error)?}
Shutdown[Stop ASR, release cam, destroy windows]
Cleanup[cleanup_resources()]
End([End])

MainLoop --> Exit
Exit -->|Yes| Shutdown
Shutdown --> Cleanup
Cleanup --> End

%% ========= STYLING =========
classDef init fill:#E6F7FF,stroke:#007ACC,stroke-width:1px;
classDef asr fill:#F9FBE7,stroke:#B0BEC5,stroke-width:1px;
classDef main fill:#E8F5E9,stroke:#66BB6A,stroke-width:1px;
classDef enroll fill:#FFF8E1,stroke:#FFA000,stroke-width:1px;
classDef cleanup fill:#F3E5F5,stroke:#8E24AA,stroke-width:1px;

class Start,Init,Audio,TTSInit,LoadPhrases,LoadDB,OpenCam,Threads,MainLoop init;
class ASRListener,A1,A2,A3,A4,A5,A6,A7,A8,A9 asr;
class MAIN,M1,M2,Activate,Deactivate,EnrollCmd,IgnoreCmd,Frame,Process,Faces,ForEach,Match,Known,Unknown,Escalate,Warn,EndLoop main;
class ENROLLMENT,E1,E2,E3,E4,E5,E6,E7,E8 enroll;
class Exit,Shutdown,Cleanup,End cleanup;

