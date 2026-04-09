import cv2
import time
import numpy as np
from uniface.detection import RetinaFace
from uniface.headpose import HeadPose
from uniface.draw import draw_head_pose

# ==================== 配置参数 ====================
# 1. 学习区域（专注状态）的角度阈值，可根据实际情况调整
FOCUS_PITCH_RANGE = (-20.0, 20.0)   # Pitch range (degrees)
FOCUS_YAW_RANGE = (-25.0, 25.0)     # Yaw range (degrees)
FOCUS_ROLL_RANGE = (-10.0, 10.0)    # Roll range (degrees)

# 2. 摄像头ID，默认为0（电脑自带摄像头）
CAMERA_ID = 0

# 3. 专注判定宽容度（秒），避免因瞬时抖动切换状态
DEBOUNCE_TIME = 1.0
# ==================== 配置结束 ====================

def is_in_focus_zone(pitch, yaw, roll):
    """Check if current head pose is within the 'study zone'"""
    return (FOCUS_PITCH_RANGE[0] < pitch < FOCUS_PITCH_RANGE[1] and
            FOCUS_YAW_RANGE[0] < yaw < FOCUS_YAW_RANGE[1] and
            FOCUS_ROLL_RANGE[0] < roll < FOCUS_ROLL_RANGE[1])

def main():
    print("[INFO] Initializing UniFace models. This may download models on first run.")
    
    # 1. Initialize models
    detector = RetinaFace()
    head_pose_estimator = HeadPose()
    
    # 2. Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera (ID: {CAMERA_ID}). Please check connection.")
        return
    
    print("[INFO] Camera opened. Press 'Q' to exit.")
    
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
        
        status_text = "No Face Detected"
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
                    
                    # 4.4 Determine focus status
                    now_in_zone = is_in_focus_zone(pitch, yaw, roll)
                    
                    # 4.5 State debounce logic
                    if now_in_zone != is_currently_focused:
                        if current_time - last_state_change_time > DEBOUNCE_TIME:
                            is_currently_focused = now_in_zone
                            last_state_change_time = current_time
                    elif is_currently_focused:
                        total_focus_time += delta_time
                    
                    # 4.6 Prepare display text
                    if is_currently_focused:
                        status_text = f"Status: Focused"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = f"Status: Distracted"
                        status_color = (0, 165, 255)  # Orange
                    
                    # 4.7 Draw head pose 3D cube (optional, can be removed for cleaner UI)
                    draw_head_pose(frame, face.bbox, pitch, yaw, roll)
                    
                    # 4.8 Draw pose data text
                    pose_info_text = f"P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}"
                    cv2.putText(frame, pose_info_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
                    
                except Exception as e:
                    status_text = f"Status: Pose Analysis Error"
                    print(f"[WARNING] Error during face processing: {e}")
        
        # 5. Draw global info panel
        # 5.1 Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1) # Adjusted size
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 5.2 Draw project title and status
        cv2.putText(frame, "Focus Island Monitor", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 5.3 Draw accumulated focus time
        mins, secs = int(total_focus_time // 60), int(total_focus_time % 60)
        time_display = f"Focused: {mins:02d}m {secs:02d}s"
        cv2.putText(frame, time_display, (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 5.4 Draw exit instruction
        cv2.putText(frame, "Press 'Q' to Exit", (frame.shape[1] - 180, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 6. Display frame
        cv2.imshow('Focus Island Monitor', frame)
        
        # 7. Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[INFO] Program exited. Total focused time: {total_focus_time:.1f} seconds.")
            break
    
    # 8. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] Program encountered an error: {e}")