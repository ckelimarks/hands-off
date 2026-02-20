#!/usr/bin/env python3
"""
Face Touch Detector - Alerts when hands touch face for > 1 second
Helps break the habit of beard pulling/face touching
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime
import pygame
import math

class FaceTouchDetector:
    def __init__(self, touch_threshold_seconds=1.0, proximity_threshold=0.08, enable_sound=True):
        """
        Args:
            touch_threshold_seconds: How long hands must touch face before alerting
            proximity_threshold: Distance threshold for "touching" (normalized 0-1)
            enable_sound: Enable annoying alarm sound
        """
        self.touch_threshold = touch_threshold_seconds
        self.proximity_threshold = proximity_threshold
        self.enable_sound = enable_sound

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.7
        )

        # Audio setup
        if self.enable_sound:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.alarm_sound = self._generate_annoying_alarm()
            self.sound_playing = False

        # State tracking
        self.touch_start_time = None
        self.is_touching = False
        self.alert_active = False

    def _generate_annoying_alarm(self):
        """Generate a really annoying alarm sound - alternating high-pitched beeps"""
        sample_rate = 22050
        duration = 0.3  # Length of each beep

        # Create an annoying alarm: rapid alternating high-pitched tones
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Two alternating frequencies (very annoying)
        freq1 = 1200  # Hz - high pitched
        freq2 = 800   # Hz - also high pitched

        # Generate alternating beeps
        beep1 = np.sin(2 * np.pi * freq1 * t[:len(t)//2])
        beep2 = np.sin(2 * np.pi * freq2 * t[len(t)//2:])
        wave = np.concatenate([beep1, beep2])

        # Add some amplitude modulation for extra annoyance
        envelope = np.abs(np.sin(2 * np.pi * 8 * t))
        wave = wave * envelope

        # Normalize and convert to 16-bit
        wave = wave * 32767 / np.max(np.abs(wave))
        wave = wave.astype(np.int16)

        # Convert to stereo
        stereo_wave = np.column_stack((wave, wave))

        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_hand_near_face(self, hand_landmarks, face_bbox):
        """Check if any hand landmarks are within the face bounding box area"""
        face_x, face_y, face_w, face_h = face_bbox

        # Expand face bbox slightly for proximity detection
        margin = self.proximity_threshold
        expanded_x = face_x - margin
        expanded_y = face_y - margin
        expanded_w = face_w + 2 * margin
        expanded_h = face_h + 2 * margin

        # Check key hand points (wrist, fingertips, palm)
        key_points = [0, 4, 8, 12, 16, 20]  # wrist + fingertips

        for idx in key_points:
            landmark = hand_landmarks.landmark[idx]
            if (expanded_x <= landmark.x <= expanded_x + expanded_w and
                expanded_y <= landmark.y <= expanded_y + expanded_h):
                return True

        return False

    def draw_alert(self, frame):
        """Draw visual alert on frame"""
        h, w = frame.shape[:2]

        # Red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Alert text
        text = "HANDS ON FACE!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 6  # Thicker for bold effect

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        # Text with outline
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                    (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                    (0, 0, 255), thickness, cv2.LINE_AA)

    def process_frame(self, frame):
        """Process a single frame and return annotated frame"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and hands
        face_results = self.face_detection.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        touching_now = False

        # If face detected
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_bbox = (bbox.xmin, bbox.ymin, bbox.width, bbox.height)

                # Draw face bounding box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h),
                             (0, 255, 0), 2)

                # Check if hands are near face
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        if self.is_hand_near_face(hand_landmarks, face_bbox):
                            touching_now = True

        # Update touch state and timing
        current_time = time.time()

        if touching_now:
            if not self.is_touching:
                # Just started touching
                self.touch_start_time = current_time
                self.is_touching = True
                self.alert_active = False
            else:
                # Still touching - check duration
                duration = current_time - self.touch_start_time
                if duration >= self.touch_threshold and not self.alert_active:
                    self.alert_active = True
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Hands on face for {duration:.1f}s")

                    # Start playing annoying sound
                    if self.enable_sound and not self.sound_playing:
                        self.alarm_sound.play(loops=-1)  # Loop indefinitely
                        self.sound_playing = True
        else:
            # Not touching
            if self.is_touching:
                duration = current_time - self.touch_start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Touch ended after {duration:.1f}s")

            # Stop sound if playing
            if self.enable_sound and self.sound_playing:
                self.alarm_sound.stop()
                self.sound_playing = False

            self.is_touching = False
            self.touch_start_time = None
            self.alert_active = False

        # Draw alert if active
        if self.alert_active:
            self.draw_alert(frame)
            duration = current_time - self.touch_start_time
            cv2.putText(frame, f"Duration: {duration:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw status info
        status = "TOUCHING" if self.is_touching else "Clear"
        color = (0, 0, 255) if self.is_touching else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame

    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Face Touch Detector started")
        print("Press 'q' to quit")
        print(f"Alert threshold: {self.touch_threshold}s")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Display
                cv2.imshow('Face Touch Detector', annotated_frame)

                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.face_detection.close()
            if self.enable_sound:
                pygame.mixer.quit()


if __name__ == "__main__":
    detector = FaceTouchDetector(
        touch_threshold_seconds=1.0,
        proximity_threshold=0.08,
        enable_sound=True  # Set to False to disable annoying alarm
    )
    detector.run()
