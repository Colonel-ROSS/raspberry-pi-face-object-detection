#!/usr/bin/env python3
import time
import csv
import cv2
import os
import datetime
import pickle
import hashlib
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import face_recognition
from collections import deque

# =========================
# Paths and cache for faces
# =========================
KNOWN_FACES_DIR = "known_faces"
CACHE_FILE = "face_encodings.pkl"
HASH_FILE = "dir_hash.txt"

def get_dir_hash(directory):
    """Compute a hash of filenames to detect changes in known_faces directory."""
    hash_obj = hashlib.md5()
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            hash_obj.update(file.encode())
    return hash_obj.hexdigest()

known_encodings, known_names = [], []

current_hash = get_dir_hash(KNOWN_FACES_DIR)
old_hash = ""
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        old_hash = f.read()

if os.path.exists(CACHE_FILE) and current_hash == old_hash:
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
    print("[INFO] Loaded encodings from cache.")
else:
    print("[INFO] Directory changed or no cache found. Generating encodings...")
    for person_folder in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person_folder)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                try:
                    img = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person_folder)
                except Exception as e:
                    print(f"[WARN] Skipped {img_path}: {e}")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

print(f"[INFO] Loaded {len(known_encodings)} encodings for {len(set(known_names))} people.")

# ======================
# Camera & model setup
# ======================
TARGET_WIDTH, TARGET_HEIGHT = 1280, 720
FPS_PLAYBACK = 1.15
PROCESS_EVERY_N_FRAMES = 3  # Process detection every 3 frames
YOLO_SIZE = 416  # Reduced from 640 for faster processing

picam2 = Picamera2()
video_config = picam2.create_preview_configuration(
    main={"size": (TARGET_WIDTH, TARGET_HEIGHT)},
    controls={"FrameRate": FPS_PLAYBACK},
)
picam2.configure(video_config)
picam2.start()

# Load YOLO model
model = YOLO("yolov5n.pt")

# ==========================
# UI window and output files
# ==========================
cv2.namedWindow("Scene Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Scene Detection", TARGET_WIDTH, TARGET_HEIGHT)

os.makedirs("scene_vids", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"scene_vids/scene_{timestamp}.mp4"

fourcc = cv2.VideoWriter_fourcc(*'avc1')
video_writer = cv2.VideoWriter(video_filename, fourcc, FPS_PLAYBACK, (TARGET_WIDTH, TARGET_HEIGHT))
if not video_writer.isOpened():
    print("[WARN] 'avc1' not available. Falling back to 'mp4v' (MPEG-4 Part 2).")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, FPS_PLAYBACK, (TARGET_WIDTH, TARGET_HEIGHT))

# ==============================
# Tracking, logging, and controls
# ==============================
active_people = {}
confidence_log = {}
log_entries = []
object_log = []
last_seen_name = {}

brightness_factor = 1.0
sharpness_factor = 1.0

# Cache for detection results
cached_detections = {
    'faces': [],
    'objects': [],
    'face_names': []
}

print("[INFO] Detection started. Press 'q' to quit.")
print("[INFO] Controls: O=Increase Brightness | P=Decrease Brightness | K=Increase Sharpness | L=Decrease Sharpness")

# =======
# Timing
# =======
frame_count = 0
processing_start = time.time()
fps_queue = deque(maxlen=30)
last_fps_time = time.time()

# =========================
# Main detection/record loop
# =========================
try:
    while True:
        loop_start = time.time()
        
        # Capture frame
        frame = picam2.capture_array()
        if frame is None:
            continue
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Only apply filters if they're not at default values (optimization)
        if brightness_factor != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
        
        if sharpness_factor != 1.0:
            sharpening_kernel = np.array([[0, -1, 0],
                                          [-1, 5 + sharpness_factor, -1],
                                          [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, sharpening_kernel)

        current_time_s = time.time() - processing_start
        annotated_frame = frame.copy()

        # ONLY process detections every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # YOLO object detection
            try:
                results = model.predict(frame, imgsz=YOLO_SIZE, verbose=False, half=True)
                cached_detections['objects'] = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    label = results[0].names[cls]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cached_detections['objects'].append((label, conf, x1, y1, x2, y2))
                    object_log.append((label, round(conf, 2), round(current_time_s, 2)))
            except Exception as e:
                print(f"[WARN] YOLO error: {e}")

            # Face recognition (on smaller image for speed)
            try:
                # Resize for face detection (significantly faster)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                
                # Scale back to original size
                face_locations = [(top*2, right*2, bottom*2, left*2) 
                                  for (top, right, bottom, left) in face_locations]
                
                cached_detections['faces'] = face_locations
                cached_detections['face_names'] = []
                
                for idx, face_encoding in enumerate(face_encodings):
                    name = "Unknown"
                    confidence = 0.0
                    if known_encodings:
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(distances)
                        if distances[best_match_index] < 0.6:  # Threshold
                            name = known_names[best_match_index]
                            confidence = round((1 - float(distances[best_match_index])), 2)
                    
                    cached_detections['face_names'].append((name, confidence))
                    
                    if name not in confidence_log:
                        confidence_log[name] = []
                    confidence_log[name].append((current_time_s, confidence))
                    
            except Exception as e:
                print(f"[WARN] Face recognition error: {e}")
                cached_detections['faces'] = []
                cached_detections['face_names'] = []

        # Draw cached detections (even on non-processing frames)
        for (label, conf, x1, y1, x2, y2) in cached_detections['objects']:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        current_names = []
        for idx, (top, right, bottom, left) in enumerate(cached_detections['faces']):
            if idx < len(cached_detections['face_names']):
                name, confidence = cached_detections['face_names'][idx]
                current_names.append(name)
                
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{name} ({confidence:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update active people (only on processing frames)
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            for name in current_names:
                if name not in active_people:
                    active_people[name] = current_time_s
                    print(f"[LOG] {name} entered at {current_time_s:.2f}s")

            for name in list(active_people.keys()):
                if name not in current_names:
                    exit_time = current_time_s
                    entries = confidence_log.get(name, [])
                    avg_conf = round(sum(c for _, c in entries) / max(len(entries), 1), 2)
                    log_entries.append((name, active_people[name], exit_time, avg_conf))
                    print(f"[LOG] {name} exited at {exit_time:.2f}s (Avg Confidence: {avg_conf})")
                    del active_people[name]

        # Display FPS
        loop_time = time.time() - loop_start
        fps_queue.append(1.0 / loop_time if loop_time > 0 else 0)
        avg_fps = sum(fps_queue) / len(fps_queue)
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show preview
        cv2.imshow("Scene Detection", annotated_frame)

        # Write frame to video
        video_writer.write(annotated_frame)

        frame_count += 1

        # Print FPS every 5 seconds
        if time.time() - last_fps_time >= 5.0:
            print(f"[INFO] Average FPS: {avg_fps:.2f}")
            last_fps_time = time.time()

        # Controls & quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Session ended.")
            break
        elif key == ord('o'):
            brightness_factor += 0.1
            print(f"[INFO] Brightness increased to {brightness_factor:.2f}")
        elif key == ord('p'):
            brightness_factor = max(0.1, brightness_factor - 0.1)
            print(f"[INFO] Brightness decreased to {brightness_factor:.2f}")
        elif key == ord('k'):
            sharpness_factor += 0.5
            print(f"[INFO] Sharpness increased to {sharpness_factor:.2f}")
        elif key == ord('l'):
            sharpness_factor = max(0.0, sharpness_factor - 0.5)
            print(f"[INFO] Sharpness decreased to {sharpness_factor:.2f}")

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    # Cleanup
    cv2.destroyAllWindows()
    video_writer.release()
    picam2.stop()
    print(f"[INFO] Video saved as {video_filename}")

    # Save movement log
    with open("movement_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Entry Time (s)", "Exit Time (s)", "Avg Confidence"])
        writer.writerows(log_entries)

    # Save object log
    with open("object_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Object", "Confidence", "Timestamp (s)"])
        writer.writerows(object_log)

    print("[INFO] Movement log saved to movement_log.csv")
    print("[INFO] Object log saved to object_log.csv")
