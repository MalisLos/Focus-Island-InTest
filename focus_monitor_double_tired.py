import cv2
import time
import numpy as np
from uniface.detection import RetinaFace
from uniface.headpose import HeadPose
from uniface.draw import draw_head_pose

# ==================== Configuration Parameters ====================
# 1. Camera ID, default is 0 (built-in webcam)
CAMERA_ID = 0

# 2. Focus state debounce time (seconds) to avoid flickering
DEBOUNCE_TIME = 1.0

# 3. Study mode definitions
# Mode 1: Computer Study (looking at screen)
COMPUTER_MODE_THRESHOLDS = {
    'name': 'Computer Study',           # Changed to English
    'pitch_range': (-20.0, 20.0),       # Pitch angle range
    'yaw_range': (-25.0, 25.0),         # Yaw angle range
    'roll_range': (-10.0, 10.0),        # Roll angle range
    'ui_color': (0, 180, 255)           # UI theme color (orange)
}

# Mode 2: Desk Study (reading/writing on desk)
DESK_MODE_THRESHOLDS = {
    'name': 'Desk Study',               # Changed to English
    'pitch_range': (-40.0, -25.0),      # Significant head-down range
    'yaw_range': (-10.0, 10.0),         # Small left-right rotation
    'roll_range': (-10.0, 10.0),        # Head tilt range
    'ui_color': (100, 200, 100)         # UI theme color (light green)
}
# ==================== Configuration End ====================

def is_in_focus_zone(pitch, yaw, roll, thresholds):
    """Check if current head pose is within the 'study zone' of specified mode"""
    return (thresholds['pitch_range'][0] < pitch < thresholds['pitch_range'][1] and
            thresholds['yaw_range'][0] < yaw < thresholds['yaw_range'][1] and
            thresholds['roll_range'][0] < roll < thresholds['roll_range'][1])

def main():
    # --- Mode Selection ---
    print("\n" + "="*40)
    print("      Focus Island - Study Mode Selection")
    print("="*40)
    print(f"  1. {COMPUTER_MODE_THRESHOLDS['name']}")
    print(f"     Facing computer screen, slight head up/down")
    print(f"  2. {DESK_MODE_THRESHOLDS['name']}")
    print(f"     Facing desk books, significant head-down for writing/reading")
    print("="*40)
    
    while True:
        try:
            choice = input("Please select study mode (enter 1 or 2): ").strip()
            if choice == '1':
                current_mode = COMPUTER_MODE_THRESHOLDS
                print(f"[INFO] Selected: {current_mode['name']} Mode")
                break
            elif choice == '2':
                current_mode = DESK_MODE_THRESHOLDS
                print(f"[INFO] Selected: {current_mode['name']} Mode")
                break
            else:
                print("[ERROR] Invalid input, please try again.")
        except KeyboardInterrupt:
            print("\n[INFO] User interrupted selection, program exiting.")
            return
    
    print("[INFO] Initializing UniFace models, may download models on first run...")
    
    # 1. Initialize models
    detector = RetinaFace()
    head_pose_estimator = HeadPose()
    
    # 2. Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera (ID: {CAMERA_ID}). Please check connection.")
        return
    
    print("[INFO] Camera started. Press 'Q' to exit program.")
    
    # 3. State variables initialization
    total_focus_time = 0.0      # Accumulated focus time (seconds)
    is_currently_focused = False # Is currently focused
    last_state_change_time = time.time() # Last state change timestamp
    last_frame_time = time.time() # For calculating delta time
    
    # 4. Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break
        
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # 4.1 Detect faces
        faces = detector.detect(frame)
        
        status_text = "No Face Detected"  # Changed to English
        status_color = (0, 0, 255)  # Red
        
        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                try:
                    # 4.3 Estimate head pose
                    result = head_pose_estimator.estimate(face_crop)
                    pitch, yaw, roll = result.pitch, result.yaw, result.roll
                    
                    # 4.4 Determine focus status (using current mode thresholds)
                    now_in_zone = is_in_focus_zone(pitch, yaw, roll, current_mode)
                    
                    # 4.5 State debounce logic
                    if now_in_zone != is_currently_focused:
                        if current_time - last_state_change_time > DEBOUNCE_TIME:
                            is_currently_focused = now_in_zone
                            last_state_change_time = current_time
                    elif is_currently_focused:
                        total_focus_time += delta_time
                    
                    # 4.6 Prepare display text
                    if is_currently_focused:
                        status_text = f"Status: Focused ({current_mode['name']})"  # Changed to English
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = f"Status: Distracted"  # Changed to English
                        status_color = (0, 165, 255)  # Orange
                    
                    # 4.7 Draw head pose 3D cube
                    draw_head_pose(frame, face.bbox, pitch, yaw, roll)
                    
                    # 4.8 Draw pose data text
                    pose_info_text = f"P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}"
                    cv2.putText(frame, pose_info_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                    
                except Exception as e:
                    status_text = f"Status: Pose Analysis Error"  # Changed to English
                    print(f"[WARNING] Error during face processing: {e}")
        
        # 5. Draw global info panel
        # 5.1 Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 5.2 Draw project title and current mode
        cv2.putText(frame, "Focus Island - Dual Mode", (20, 40),  # Changed to English
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        mode_display = f"Mode: {current_mode['name']}"  # Changed to English
        cv2.putText(frame, mode_display, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_mode['ui_color'], 2)
        
        # 5.3 Draw focus status
        cv2.putText(frame, status_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 5.4 Draw accumulated focus time
        mins, secs = int(total_focus_time // 60), int(total_focus_time % 60)
        time_display = f"Focus Time: {mins:02d}m {secs:02d}s"  # Changed to English
        cv2.putText(frame, time_display, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5.5 Draw exit instruction
        cv2.putText(frame, "Press 'Q' to Exit", (frame.shape[1] - 150, frame.shape[0] - 20),  # Changed to English
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 6. Display frame
        cv2.imshow('Focus Island - Dual Mode', frame)
        
        # 7. Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[INFO] Program exited. Total focus time: {total_focus_time:.1f} seconds.")
            break
    
    # 8. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] Program encountered an error: {e}")