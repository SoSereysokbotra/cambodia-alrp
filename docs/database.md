-- Create registered_plates table
CREATE TABLE IF NOT EXISTS registered_plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT UNIQUE NOT NULL,
    owner_name TEXT NOT NULL,
    vehicle_type TEXT,
    registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'expired')),
    notes TEXT
);

-- Create plate_reads table
CREATE TABLE IF NOT EXISTS plate_reads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_text TEXT,
    detected_plate TEXT NOT NULL,
    yolo_confidence REAL CHECK (yolo_confidence >= 0.0 AND yolo_confidence <= 1.0),
    crnn_confidence REAL CHECK (crnn_confidence >= 0.0 AND crnn_confidence <= 1.0),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location TEXT DEFAULT 'Main Gate',
    action TEXT CHECK (action IN ('ENTRY_ALLOWED', 'ENTRY_DENIED', 'REVIEW_REQUIRED', 'MANUAL_OVERRIDE', 'ERROR')),
    photo_path TEXT
);

-- Create system_metrics table (optional)
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fps REAL,
    avg_latency_ms REAL,
    gpu_memory_mb REAL,
    cpu_usage_percent REAL,
    rtsp_connected INTEGER,
    total_detections_today INTEGER,
    uptime_percent REAL
);