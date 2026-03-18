import cv2
import mediapipe as mp
import numpy as np
import math

class PostureEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Use complexity 0 for faster performance in web-streaming scenarios
        self.pose_front = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
        self.pose_side = self.mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)

    def calculate_angle(self, a, b, c):
        """Calculates the angle at point B given three points A, B, C."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def get_vertical_angle(self, p_base, p_target):
        """Calculates the angle of a vector relative to the vertical axis."""
        dx = p_target.x - p_base.x
        dy = p_target.y - p_base.y
        # Vertical axis is (0, -1) relative to the base
        angle = math.degrees(math.atan2(dx, -dy))
        return angle

    def process_frames(self, frame_f, frame_s):
        # 1. Process Front View (Shoulder Alignment)
        results_f = self.pose_front.process(cv2.cvtColor(frame_f, cv2.COLOR_BGR2RGB))
        sh_diff = 0
        if results_f.pose_landmarks:
            lm = results_f.pose_landmarks.landmark
            # Points 11 (L) and 12 (R)
            h, w, _ = frame_f.shape
            sh_diff = abs(lm[11].y - lm[12].y) * h
            
            # Draw Front 6-Point Skeleton
            for idx in [11, 12, 23, 24]:
                cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame_f, (cx, cy), 8, (0, 255, 0), -1)
            cv2.line(frame_f, (int(lm[11].x*w), int(lm[11].y*h)), (int(lm[12].x*w), int(lm[12].y*h)), (0, 255, 0), 2)

        # 2. Process Side View (Neck & Spine)
        results_s = self.pose_side.process(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB))
        neck_tilt = 0
        spine_angle = 0
        if results_s.pose_landmarks:
            lm = results_s.pose_landmarks.landmark
            h, w, _ = frame_s.shape
            
            # Neck Tilt: Ear (7) to Shoulder (11) relative to vertical
            neck_tilt = self.get_vertical_angle(lm[11], lm[7])
            
            # Spine Angle: Shoulder (11) to Hip (23) relative to vertical
            spine_angle = self.get_vertical_angle(lm[23], lm[11])
            
            # Draw Side 6-Point Skeleton
            for idx in [7, 11, 23]:
                cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame_s, (cx, cy), 8, (255, 50, 50), -1)
            # Draw lines connecting the 6-point side profile
            cv2.line(frame_s, (int(lm[7].x*w), int(lm[7].y*h)), (int(lm[11].x*w), int(lm[11].y*h)), (255, 255, 255), 2)
            cv2.line(frame_s, (int(lm[11].x*w), int(lm[11].y*h)), (int(lm[23].x*w), int(lm[23].y*h)), (255, 255, 255), 2)

        metrics = {
            "shoulder_diff": round(sh_diff, 1),
            "neck_tilt": round(neck_tilt, 1),
            "spine_angle": round(spine_angle, 1)
        }
        
        return frame_f, frame_s, metrics

# --- TEST SCRIPT (To run without Flask) ---
if __name__ == "__main__":
    engine = PostureEngine()
    cap_f = cv2.VideoCapture(0)
    cap_s = cv2.VideoCapture(1)

    while True:
        ret_f, f_f = cap_f.read()
        ret_s, f_s = cap_s.read()
        if not ret_f or not ret_s: break

        proc_f, proc_s, data = engine.process_frames(f_f, f_s)
        
        # Display Text Overlay
        cv2.putText(proc_f, f"Sh. Diff: {data['shoulder_diff']}", (10, 30), 1, 1, (0,255,0), 2)
        cv2.putText(proc_s, f"Neck: {data['neck_tilt']} / Spine: {data['spine_angle']}", (10, 30), 1, 1, (255,50,50), 2)

        combined = np.hstack((cv2.resize(proc_f, (640, 480)), cv2.resize(proc_s, (640, 480))))
        cv2.imshow("6-Point Posture AI", combined)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap_f.release()
    cap_s.release()
    cv2.destroyAllWindows()