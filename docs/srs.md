# Software Requirements Specification (SRS)
## Cambodian Automatic License Plate Recognition System
### Student Deep Learning Project - Proof of Concept

**Version:** 2.0 (Student Edition)  
**Date:** January 2025  
**Status:** Development Phase - Approved for Implementation  
**Project Duration:** 12–16 weeks  
**Organization:** Kirirom Institute of Technology (Year 2, Deep Learning Course)  
**Project Type:** Academic capstone project with PoC pilot deployment

---

## Table of Contents

1. Introduction
2. Overall Description
3. Specific Functional Requirements
4. Specific Non-Functional Requirements
5. System Features
6. External Interface Requirements
7. System Constraints
8. Performance Requirements
9. Security Requirements
10. Acceptance Criteria
11. Appendices

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification document defines the functional and non-functional requirements for the Cambodian Automatic License Plate Recognition (ALPR) System as a student deep learning capstone project. The system demonstrates the integration of computer vision (YOLOv10 detection + CRNN recognition) with real-time video processing and automated decision-making on edge hardware.

The goal is not production deployment at scale, but rather to build a working prototype that can operate in a real environment (parking lot or gated community) for a limited pilot duration (2 weeks), demonstrating practical application of deep learning models.

### 1.2 Scope

The ALPR system scope includes:

**In Scope:**
- Real-time license plate detection using YOLOv10 deep learning model
- Khmer and Latin character recognition using CRNN with CTC loss
- Vehicle whitelist management (50–100 authorized plates)
- Automated gate control via MQTT protocol (simulated or real)
- **Real-time video stream from smartphone acting as IP camera (Wi-Fi streaming)**
- Pilot deployment at one location for 2 weeks (proof of concept)
- Edge computing infrastructure (laptop or Jetson Nano)
- Comprehensive audit logging with photos

**Out of Scope:**
- Multi-camera deployment (single camera/smartphone)
- Mobile app for residents
- API integrations with external systems
- Geographic expansion beyond pilot location
- Payment or billing system
- Hardware manufacturing or installation service
- Cloud-based architecture (local network only)

### 1.3 Definitions, Acronyms, Abbreviations

- **ALPR:** Automatic License Plate Recognition
- **CRNN:** Convolutional Recurrent Neural Network
- **CTC:** Connectionist Temporal Classification
- **mAP:** mean Average Precision
- **CER:** Character Error Rate
- **SER:** Sequence Error Rate
- **RTSP:** Real Time Streaming Protocol
- **MQTT:** Message Queuing Telemetry Transport
- **IoU:** Intersection over Union
- **FPS:** Frames Per Second
- **GPU:** Graphics Processing Unit
- **Edge PC:** Computing device (laptop with GPU or Jetson Nano)
- **Khmer:** Cambodian script/language
- **Smartphone IP Camera:** A smartphone streaming video over Wi-Fi using apps like IP Webcam, Iriun, or DroidCam
- **Pilot:** Initial 2-week live deployment test at one location
- **PoC:** Proof of Concept

### 1.4 References

- YOLOv10 Documentation: https://github.com/THU-MIX/yolov10
- CRNN Research Paper: CRNN for Scene Text Recognition (Shi et al., 2016)
- MQTT Specification: https://mqtt.org/
- Roboflow Annotation Tool: https://roboflow.com
- IP Webcam (Android): https://www.ip-webcam.appspot.com/
- Iriun Webcam: https://iriun.com/

### 1.5 Document Overview

This SRS is organized into 11 main sections covering all aspects of the student ALPR project, from functional requirements through acceptance criteria. Each requirement is uniquely identified for traceability.

---

## 2. Overall Description

### 2.1 Product Perspective

The ALPR system operates as a proof-of-concept gate automation solution for academic demonstration. The system interfaces with:

- **Input:** Smartphone (Android/iOS) streaming video via Wi-Fi using RTSP or HTTP streaming
- **Processing:** Laptop or Jetson Nano running YOLOv10 detection and CRNN recognition models
- **Data Store:** SQLite database containing registered plates and audit logs (local storage)
- **Output:** MQTT commands to simulated gate system or hardware relay (ESP32 optional)
- **User:** Building security or project demonstrator and authorized residents (for pilot)

The system is designed to operate continuously for 2 weeks with manual oversight and exception handling.

### 2.2 Product Functions

**Primary Functions:**

1. **Video Capture:** Acquire continuous video stream from smartphone over Wi-Fi
2. **Plate Detection:** Identify license plate regions in video frames using YOLOv10
3. **Character Recognition:** Extract and recognize characters from detected plates using CRNN
4. **Database Query:** Verify detected plates against registered vehicle whitelist
5. **Gate Control:** Send activation commands to gate system via MQTT (or simulation)
6. **Audit Logging:** Record all detection events with photos and metadata
7. **System Monitoring:** Track performance metrics and alert on anomalies

**Secondary Functions:**

8. **Manual Override:** Allow operator to manually control gate
9. **Low Confidence Handling:** Flag uncertain detections for human review
10. **Historical Search:** Query audit logs by date, plate, or vehicle
11. **Administrative Interface:** Register/deregister vehicles for testing

### 2.3 User Characteristics

**Primary Users:**

- **Project Demonstrator/Operator:** Student/researcher monitoring the gate system during pilot
  - Technical skill: Advanced (student developer)
  - Frequency: Continuous during pilot (2 weeks)
  - Key needs: System visibility, debugging capabilities, manual controls

- **Test Manager:** Manages vehicle whitelist and collects pilot data
  - Technical skill: Moderate
  - Frequency: Daily during pilot
  - Key needs: Easy vehicle management, audit trail review

**Secondary Users:**

- **Residents/Test Vehicles:** Vehicle owners whose plates are pre-registered for pilot
  - Technical skill: None required (system operates transparently)
  - Frequency: Multiple entries per day
  - Key needs: Fast gate opening, zero false denials during pilot

### 2.4 Operating Environment

**Hardware Environment:**

- **Edge PC:** Laptop with GPU (NVIDIA RTX 2060 / 3060 / RTX 4050) or NVIDIA Jetson Nano/Orin Nano
- **Smartphone:** Any modern Android (API 21+) or iOS 12+ with good camera
- **Gate Control:** Optional ESP32 with relay module (can be simulated in software)
- **Storage:** Local SSD or HDD (minimum 32GB)
- **Network:** Local Wi-Fi network with <100m range from smartphone to edge PC

**Software Environment:**

- **Operating System:** Ubuntu 20.04/22.04 LTS (on laptop) or Jetson OS
- **Runtime:** Python 3.9+
- **Deep Learning Framework:** PyTorch 2.0+
- **Database:** SQLite 3 (no external database)
- **Message Broker:** Mosquitto (MQTT broker, optional for gate simulation)
- **Video Processing:** OpenCV 4.5+
- **Smartphone Streaming:** IP Webcam app (Android) or equivalent

**Network Environment:**

- Local Wi-Fi network (no external internet required)
- Smartphone accessible via IP camera app (RTSP or HTTP stream)
- MQTT broker on edge PC (if used)
- No cloud connectivity required

### 2.5 General Constraints

- **Accuracy:** System should achieve YOLOv10 mAP ≥ 0.80 and CRNN CER ≤ 10% on validation set
- **Latency:** Total detection-to-gate-command time should be <500ms
- **Availability:** System should operate continuously for 14 days without crashes
- **Localization:** System must recognize Cambodian Khmer script
- **Cost:** Total project hardware cost <$500 (leverages existing smartphone/laptop)
- **Power:** System runs on standard AC power (UPS optional for edge device)
- **Compliance:** Audit logs retained for pilot duration; deleted post-project

### 2.6 Assumptions and Dependencies

**Assumptions:**

- Smartphone camera has good enough quality for plate recognition (720p+ or equivalent)
- Wi-Fi network is stable and indoor range sufficient (<100m)
- Project team provides accurate list of 50–100 authorized plates before pilot
- Smartphone can be mounted on fixed position near gate
- Building manager/owner permits limited pilot testing

**Dependencies:**

- YOLOv10 pretrained weights (COCO dataset)
- CRNN implementation from open-source community (available on GitHub)
- MQTT protocol (Mosquitto available free)
- Smartphone IP camera app availability (IP Webcam free, Iriun free tier)
- Roboflow platform for dataset annotation (free tier available)
- GPU availability for model training (Google Colab, local GPU, or university lab)

---

## 3. Specific Functional Requirements

### 3.1 Video Input & Streaming (VID-xxx)

**VID-001: Smartphone IP Camera Connection**
- The system SHALL connect to the smartphone camera via RTSP or HTTP streaming protocol
- The system SHALL automatically reconnect if stream drops (retry every 5 seconds, max 10 retries)
- The system SHALL decode video frames at minimum 15 FPS (real smartphone streams typically 20–30 FPS)
- Priority: HIGH | Status: REQUIRED

**VID-002: Frame Extraction**
- The system SHALL extract individual frames from the video stream
- The system SHALL drop frames if GPU processing cannot keep pace (maintain queue size <5)
- The system SHALL timestamp each frame with acquisition time (precision: milliseconds)
- Priority: HIGH | Status: REQUIRED

**VID-003: Video Format Support**
- The system SHALL support H.264 video codec (standard for smartphone streaming)
- The system SHALL support frame resolutions from 720p to 1080p
- The system SHALL handle variable frame rates (15–30 FPS)
- Priority: MEDIUM | Status: REQUIRED

**VID-004: Smartphone Camera Positioning**
- Smartphone SHOULD be mounted at 30–45° angle to capture frontal/side plate views
- Smartphone SHOULD be positioned 2–3 meters from vehicle entry point
- Smartphone camera should have clear, unobstructed view of license plates
- Priority: MEDIUM | Status: REQUIRED (setup guideline)

### 3.2 Plate Detection (DET-xxx)

**DET-001: YOLOv10 Model Inference**
- The system SHALL use YOLOv10 deep learning model for plate detection
- The system SHALL achieve mean Average Precision (mAP) >= 0.80 at IoU=0.5 on validation set
- The system SHALL run inference on available GPU with latency <50ms per frame
- Priority: CRITICAL | Status: REQUIRED

**DET-002: Bounding Box Output**
- The system SHALL output bounding box coordinates for each detected plate
- Bounding boxes SHALL be in normalized coordinates (0.0–1.0) relative to frame size
- The system SHALL output confidence score for each detection (0.0–1.0)
- Priority: HIGH | Status: REQUIRED

**DET-003: Detection Confidence Threshold**
- The system SHALL accept detections with confidence >= 0.50
- The system SHALL ignore detections with confidence < 0.50
- Priority: HIGH | Status: REQUIRED

**DET-004: Multiple Plate Handling**
- The system SHALL handle frames containing multiple vehicles
- The system SHALL process all detections or top 3 if many plates present
- Priority: MEDIUM | Status: REQUIRED

**DET-005: Plate Crop Extraction**
- The system SHALL extract plate region from frame using bounding box
- The system SHALL add 10% padding around bounding box for context
- The system SHALL resize crop to 64x320 pixels for CRNN input
- Priority: HIGH | Status: REQUIRED

### 3.3 Character Recognition (REC-xxx)

**REC-001: CRNN Model Inference**
- The system SHALL use CRNN with bidirectional LSTM for character recognition
- The system SHALL achieve Character Error Rate (CER) <= 10% on validation set
- The system SHALL run inference on GPU with latency <100ms per plate crop
- Priority: CRITICAL | Status: REQUIRED

**REC-002: Character Set Support**
- The system SHALL recognize Khmer base consonants (33 characters)
- The system SHALL recognize Khmer vowel diacritics (12 characters)
- The system SHALL recognize Latin uppercase letters (A–Z, 26 characters)
- The system SHALL recognize numeric digits (0–9, 10 characters)
- Total character set: 50+ unique characters
- Priority: CRITICAL | Status: REQUIRED

**REC-003: Variable-Length Output**
- The system SHALL handle plate texts of variable length (7–10 characters typical)
- The system SHALL use CTC (Connectionist Temporal Classification) loss for alignment
- Priority: HIGH | Status: REQUIRED

**REC-004: Confidence Scoring**
- The system SHALL output per-character confidence scores
- The system SHALL output overall plate recognition confidence (0.0–1.0)
- Overall confidence calculation: average of per-character confidences
- Priority: HIGH | Status: REQUIRED

**REC-005: Low Confidence Handling**
- The system SHALL flag recognized plates with confidence < 0.70 for manual review
- The system SHALL still log these results but with "REVIEW_REQUIRED" action
- The system SHALL NOT automatically open gate for low-confidence plates
- Priority: CRITICAL | Status: REQUIRED

**REC-006: Khmer Text Normalization**
- The system SHALL normalize recognized Khmer text (remove extra spaces, standardize diacritics)
- The system SHALL support plates in format "ProvinceName NNNXXX" (Khmer + Latin)
- Priority: MEDIUM | Status: REQUIRED

### 3.4 Database Management (DB-xxx)

**DB-001: Registered Plates Table**
- The system SHALL maintain a SQLite table "registered_plates" with fields:
  - id (INTEGER PRIMARY KEY)
  - plate_text (TEXT UNIQUE)
  - owner_name (TEXT)
  - vehicle_type (TEXT)
  - registered_date (TIMESTAMP)
  - status (TEXT: "active", "suspended")
  - notes (TEXT)
- Priority: CRITICAL | Status: REQUIRED

**DB-002: Plate Reads Audit Log**
- The system SHALL maintain a SQLite table "plate_reads" logging every detection:
  - id, plate_text, detected_plate, yolo_confidence, crnn_confidence
  - timestamp, location, action, photo_path
- Priority: CRITICAL | Status: REQUIRED

**DB-003: Data Persistence**
- The system SHALL persist all records to local SSD/HDD with ACID guarantees
- Priority: HIGH | Status: REQUIRED

**DB-004: Whitelist Query**
- The system SHALL query registered_plates by plate_text with exact match
- Query response time: <10ms
- Priority: CRITICAL | Status: REQUIRED

**DB-005: Audit Log Retention**
- The system SHALL retain audit logs for entire pilot duration (14 days minimum)
- Logs can be deleted after project completion
- Priority: MEDIUM | Status: REQUIRED

### 3.5 Gate Control & MQTT (GTC-xxx)

**GTC-001: MQTT Broker Setup**
- The system SHALL run Mosquitto MQTT broker on edge PC (localhost:1883) or use simulation
- Priority: MEDIUM | Status: REQUIRED

**GTC-002: MQTT Publishing**
- The system SHALL publish gate control commands to topic "gate/control"
- Payload format: Plain text "GATE_OPEN" or "GATE_CLOSE"
- Priority: CRITICAL | Status: REQUIRED

**GTC-003: Gate Open Command**
- The system SHALL publish "GATE_OPEN" when:
  - Plate detected, recognized, and in whitelist with status="active"
- Priority: CRITICAL | Status: REQUIRED

**GTC-004: Gate Close Command**
- The system SHALL publish "GATE_CLOSE" when:
  - Plate not detected, not recognized, or not in whitelist
- Priority: CRITICAL | Status: REQUIRED

**GTC-005: Simulation Mode**
- The system SHALL support simulation mode (no physical gate relay required)
- In simulation, log all decisions without hardware activation
- Priority: HIGH | Status: REQUIRED

### 3.6 Logging & Audit Trail (LOG-xxx)

**LOG-001: Event Logging**
- The system SHALL log every plate detection event with:
  - Timestamp, detected plate text, confidence scores, gate action, vehicle photo
- Priority: CRITICAL | Status: REQUIRED

**LOG-002: Photo Capture**
- The system SHALL capture a photo when plate is successfully detected/recognized
- Photo format: JPEG, minimum 720p resolution
- Storage: Local file system with naming convention "plate_TIMESTAMP_PLATE.jpg"
- Priority: HIGH | Status: REQUIRED

**LOG-003: Error Logging**
- The system SHALL log all errors with timestamp, message, and severity level
- Error log file: /var/log/alpr.log (or project directory)
- Priority: HIGH | Status: REQUIRED

### 3.7 Manual Override (MAN-xxx)

**MAN-001: Manual Gate Control**
- The system SHALL provide a keyboard command or button to manually open gate
- Manual open shall be logged with timestamp
- Priority: HIGH | Status: REQUIRED

**MAN-002: Emergency Stop**
- The system SHALL support emergency stop to immediately close gate
- E-Stop logs emergency event with timestamp
- Priority: CRITICAL | Status: REQUIRED

### 3.8 Administrative Functions (ADM-xxx)

**ADM-001: Vehicle Registration**
- The system SHALL allow adding new vehicles to whitelist (manual entry)
- Input: plate_text, owner_name, vehicle_type
- Priority: HIGH | Status: REQUIRED

**ADM-002: Vehicle Deregistration**
- The system SHALL allow suspending vehicles (set status to "suspended")
- Priority: HIGH | Status: REQUIRED

**ADM-003: Audit Log Search**
- The system SHALL support searching by date range, plate text, or action type
- Priority: MEDIUM | Status: REQUIRED

### 3.9 System Health & Monitoring (HLT-xxx)

**HLT-001: System Uptime Tracking**
- The system SHALL track uptime percentage (target: >=98% during pilot)
- Priority: MEDIUM | Status: REQUIRED

**HLT-002: Performance Metrics Collection**
- The system SHALL collect per-frame metrics: latency, FPS, GPU memory, CPU usage
- Metrics logged hourly
- Priority: MEDIUM | Status: REQUIRED

**HLT-003: Alert Generation**
- The system SHALL generate alerts for:
  - Smartphone stream disconnection (timeout >15s)
  - GPU out-of-memory
  - Database errors
  - Inference latency >500ms
- Priority: HIGH | Status: REQUIRED

---

## 4. Specific Non-Functional Requirements

### 4.1 Performance Requirements (PERF-xxx)

**PERF-001: End-to-End Latency**
- Total latency from plate visible in frame to gate decision: <500ms (acceptable for PoC)
- Priority: MEDIUM | Status: REQUIRED

**PERF-002: Frames Per Second (FPS)**
- System SHALL process minimum 10 FPS from smartphone stream
- Target: 15–20 FPS for continuous coverage
- Priority: HIGH | Status: REQUIRED

**PERF-003: Model Inference Speed**
- YOLOv10 inference: <50ms on edge device (laptop or Jetson)
- CRNN inference: <100ms on edge device
- Priority: CRITICAL | Status: REQUIRED

**PERF-004: Memory Footprint**
- Total system RAM usage: <6GB (including OS, models, and video processing)
- Priority: MEDIUM | Status: REQUIRED

### 4.2 Availability & Reliability (AVAIL-xxx)

**AVAIL-001: System Availability**
- Target uptime during pilot: >=98% (max 20 minutes downtime over 14 days)
- Priority: HIGH | Status: REQUIRED

**AVAIL-002: Graceful Degradation**
- If smartphone stream drops, system SHALL attempt reconnection with visual alert
- If GPU fails, system SHALL alert operator (manual gate control available)
- Priority: HIGH | Status: REQUIRED

**AVAIL-003: Automatic Recovery**
- Upon smartphone stream disconnect: Auto-reconnect every 5 seconds
- Upon transient GPU error: Retry inference up to 3 times
- Priority: HIGH | Status: REQUIRED

### 4.3 Security Requirements (SEC-xxx)

**SEC-001: Access Control**
- System SHALL allow ONLY registered plates to open gate
- False positive (unauthorized entry): Unacceptable; must be zero during pilot
- Priority: CRITICAL | Status: REQUIRED

**SEC-002: Exact Plate Matching**
- System SHALL match detected plate text EXACTLY with registered_plates
- No fuzzy/partial matching
- Priority: CRITICAL | Status: REQUIRED

**SEC-003: Audit Trail Integrity**
- ALL gate open/close events SHALL be logged with photos
- Priority: CRITICAL | Status: REQUIRED

**SEC-004: Network Isolation**
- System SHALL operate on local Wi-Fi network only
- No cloud connectivity required
- Priority: HIGH | Status: REQUIRED

**SEC-005: Fail-Safe Design**
- Upon critical error, gate SHALL default to CLOSED
- Gate SHALL NEVER remain open indefinitely
- Priority: CRITICAL | Status: REQUIRED

### 4.4 Usability Requirements (USAB-xxx)

**USAB-001: User Interface Simplicity**
- Project demonstrator SHALL operate system with minimal training
- Manual controls clearly labeled and accessible
- Priority: MEDIUM | Status: REQUIRED

**USAB-002: System Status Display**
- System SHALL show: Current frame, detected plates, gate status, last 10 events
- Priority: MEDIUM | Status: REQUIRED

### 4.5 Maintainability Requirements (MAINT-xxx)

**MAINT-001: Code Documentation**
- All functions/modules SHALL have docstrings
- Complex algorithms SHALL have inline comments
- Priority: HIGH | Status: REQUIRED

**MAINT-002: Configuration Files**
- System configuration (thresholds, model paths) in config.yaml
- Changes don't require code recompilation
- Priority: MEDIUM | Status: REQUIRED

### 4.6 Portability & Compatibility (PORT-xxx)

**PORT-001: Hardware Compatibility**
- System SHALL run on laptop with NVIDIA GPU (RTX 2060+)
- System SHALL also support Jetson Nano or equivalent ARM GPU
- Priority: HIGH | Status: REQUIRED

**PORT-002: Operating System**
- Primary: Ubuntu 20.04/22.04 LTS
- Priority: HIGH | Status: REQUIRED

**PORT-003: Python Version**
- Minimum: Python 3.9
- Priority: MEDIUM | Status: REQUIRED

---

## 5. System Features

### Feature F1: Smartphone Video Streaming

**Description:** Real-time video acquisition from smartphone over Wi-Fi

**Implementation:**
- Smartphone streams via IP Webcam (Android) or Iriun (iOS)
- RTSP endpoint accessible on local network
- Example: `rtsp://192.168.1.5:8554/h264_pcm.sdp` (IP Webcam)

**Acceptance Criteria:**
- Smartphone app successfully starts streaming
- Edge PC connects and receives frames at 20+ FPS
- Stream persists for 2+ hours without disconnection

---

### Feature F2: Plate Detection with YOLOv10

**Description:** Real-time license plate localization using deep learning

**Acceptance Criteria:**
- mAP >= 0.80 on validation dataset
- Latency <50ms per frame on target hardware
- Handles 1–3 plates per frame

---

### Feature F3: Character Recognition with CRNN

**Description:** OCR for Khmer and Latin characters on detected plates

**Acceptance Criteria:**
- CER <= 10% on validation set
- Latency <100ms per plate crop
- Khmer diacritics correctly recognized

---

### Feature F4: Vehicle Whitelist Management

**Description:** Track authorized vehicles and manage database

**Acceptance Criteria:**
- Add vehicle: <2 seconds
- Query whitelist: <10ms response time
- Export to CSV on demand

---

### Feature F5: Automated Gate Control

**Description:** Grant/deny access based on plate registration

**Acceptance Criteria:**
- Registered vehicles open gate automatically
- Unregistered vehicles denied entry
- Zero false authorizations during pilot
- <2% false denials (authorized denied entry)

---

### Feature F6: Audit Logging & Photos

**Description:** Complete record of all access attempts

**Acceptance Criteria:**
- 100% of detections logged
- Photos captured for successful reads
- Logs searchable by date/plate/action

---

### Feature F7: Manual Override & E-Stop

**Description:** Operator controls for emergency situations

**Acceptance Criteria:**
- Manual open works within 1 second
- Emergency stop closes gate immediately
- All actions logged

---

### Feature F8: System Monitoring & Alerts

**Description:** Real-time performance tracking

**Acceptance Criteria:**
- Uptime tracking accurate
- Latency metrics collected hourly
- Alerts triggered within 30 seconds of errors

---

## 6. External Interface Requirements

### 6.1 User Interfaces

**UI-001: Control Dashboard**

Display during operation:
- Live camera frame with detected plates highlighted
- Current gate status (open/closed)
- Last 20 detection events with timestamps
- System metrics (FPS, latency, GPU memory)
- Manual "Open Gate" and "E-Stop" buttons

Platform: Terminal-based GUI (PySimpleGUI or similar) or web dashboard (Flask optional)

### 6.2 Hardware Interfaces

**HW-001: Smartphone IP Camera Interface**

Protocol: RTSP or HTTP streaming
Stream URL: Configured in config.yaml
Resolution: 720p–1080p
Frame rate: 20–30 FPS

**HW-002: MQTT Interface (Optional)**

Broker: Mosquitto on localhost:1883
Topic: gate/control
Payload: "GATE_OPEN" or "GATE_CLOSE"

**HW-003: Storage Interface**

Database: SQLite at ./data/alpr.db
Photos: ./photos/ directory
Logs: ./logs/alpr.log

### 6.3 Software Interfaces

**SW-001: YOLOv10 Model**

Input: Image array (numpy, shape [H, W, 3], uint8 0–255)
Output: Detections with boxes, confidences, class IDs
Latency: <50ms inference

**SW-002: CRNN Model**

Input: Image tensor (torch, shape [1, 3, 64, 320], float32)
Output: Text string + per-character confidences
Latency: <100ms inference

**SW-003: SQLite Database**

Schema: registered_plates, plate_reads tables
Query timeout: 10 seconds max
Concurrent access: Read-safe, write-atomic

---

## 7. System Constraints

### 7.1 Hardware Constraints

- GPU VRAM: Minimum 2GB, recommended 4GB
- System RAM: Minimum 4GB, recommended 8GB
- Storage: 32GB minimum for project duration
- Network: Local Wi-Fi (no external internet required)

### 7.2 Software Constraints

- OS: Ubuntu 20.04/22.04 LTS
- Python: 3.9+
- PyTorch: 2.0+
- CUDA: 11.8+ (if using NVIDIA GPU)

### 7.3 Operational Constraints

- Pilot duration: 14 days continuous operation
- Smartphone must remain powered and Wi-Fi connected during pilot
- Manual monitoring required (no fully autonomous operation)
- Building access and gate control required for testing

### 7.4 Regulatory Constraints

- Photo retention: Limited to pilot duration only
- Data privacy: Photos not shared without permission
- No external systems accessed (local network only)

---

## 8. Performance Requirements

| Metric | Target | Acceptance |
|--------|--------|-----------|
| YOLOv10 mAP | ≥ 0.80 | Week 8 validation |
| CRNN CER | ≤ 10% | Week 8 validation |
| End-to-end latency | < 500ms | 95th percentile |
| FPS processing | ≥ 15 FPS | Continuous |
| System uptime | ≥ 98% | During 14-day pilot |
| False positives | 0 | Critical requirement |
| False negatives | < 2% | Acceptable for PoC |

---

## 9. Security Requirements

- Exact plate matching (no fuzzy logic)
- Low-confidence predictions require manual review
- All gate operations logged with photos
- Fail-safe default: gate closed on error
- Local network only (no external connectivity)

---

## 10. Acceptance Criteria

### Phase 1 PoC (Week 3)

- [ ] YOLOv10 trained, mAP 0.60–0.75 on 100 plates
- [ ] CRNN trained, CER 15–25% on 100 plates
- [ ] End-to-end latency < 1000ms average
- [ ] Zero crashes on test images

### Phase 2 Refinement (Week 8)

- [ ] mAP ≥ 0.80 on 2,000-image dataset
- [ ] CER ≤ 10% on 2,000 plates
- [ ] Error analysis report completed
- [ ] Models ready for hardware testing

### Phase 3 Integration (Week 12)

- [ ] Smartphone streaming validated (20+ FPS)
- [ ] Database schema created and tested
- [ ] MQTT integration working
- [ ] End-to-end latency < 500ms

### Phase 4 Pilot (Week 14)

- [ ] ≥ 95% successful plate reads (Week 13 soft launch)
- [ ] 0 false positives (Week 14 automation)
- [ ] < 2% false negatives
- [ ] ≥ 98% uptime during pilot
- [ ] All logs and photos successfully saved

---

## 11. Appendices

### A. Data Dictionary

**registered_plates table:**
- id (PK), plate_text (UNIQUE), owner_name, vehicle_type, registered_date, status, notes

**plate_reads table:**
- id (PK), plate_text, detected_plate, yolo_confidence, crnn_confidence, timestamp, location, action, photo_path

### B. Model Specifications

**YOLOv10:** Nano or Small variant, COCO pretrained weights

**CRNN:** CNN + Bi-LSTM + CTC, 50+ character set (Khmer + Latin)

### C. Configuration Example

```yaml
# config.yaml
smartphone:
  rtsp_url: "rtsp://192.168.1.5:8554/h264_pcm.sdp"
  reconnect_interval: 5  # seconds
  frame_timeout: 15     # seconds

yolo:
  model_path: "models/yolov10s.pt"
  confidence_threshold: 0.5
  device: "cuda"

crnn:
  model_path: "models/crnn_final.pth"
  confidence_threshold: 0.8
  device: "cuda"

database:
  path: "data/alpr.db"

mqtt:
  broker_host: "localhost"
  broker_port: 1883
  topic: "gate/control"

logging:
  photo_directory: "photos/"
  log_file: "logs/alpr.log"
```

### D. Deployment Checklist

**Pre-Pilot:**
- [ ] Smartphone IP camera app installed and streaming
- [ ] Edge PC connected to same Wi-Fi network
- [ ] YOLOv10 and CRNN models downloaded
- [ ] SQLite database created and empty
- [ ] 50–100 authorized plates pre-loaded
- [ ] Smartphone mounted at optimal angle
- [ ] Manual override procedure documented
- [ ] Operator trained on system use

**Go-Live (Week 13):**
- [ ] Logging-only mode enabled
- [ ] System operational and logging events
- [ ] Daily metrics review

**Live Automation (Week 14):**
- [ ] After Week 13 validation, enable gate automation
- [ ] Continuous monitoring for failures

---

**End of SRS Document**

**Version:** 2.0 (Student Edition)  
**Date:** January 2025  
**Status:** Development Phase