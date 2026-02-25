"""
dummy.py

Live-Pose Erkennung mit MediaPipe + OpenCV.
Zeichnet ein sauberes Strichmännchen (Stickman) über die erkannte Pose.

Drücke 'q' zum Beenden.
"""

import cv2
import mediapipe as mp
import time
from collections import deque

# ------- Einstellungen -------
WEBCAM_INDEX = 0             # anpassen bei mehrern angeschlossenen Kameras 
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SMOOTHING_WINDOW = 4         # Glättung der Landmark-Positionen (kleiner = weniger Verzögerung)
SHOW_FPS = True
# -----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Connections (paarweise Landmark-Indizes) die wir als "Strich" zeichnen
# Wir nutzen MediaPipe Pose-Landmarks (siehe mediapipe docs) - Indizes sind konstant.
# Link-Referenz: mp_pose.PoseLandmark.<NAME>.value
POSE_CONNECTIONS = [
    # Rumpf
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    # Arme
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    # Beine
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
    # Hals / Kopf
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.NOSE.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.NOSE.value),
]

# Optionale hübsche Kreise für Gelenke (als "Knöpfe" am Strichmann)
JOINTS_TO_DRAW = {
    'head': [mp_pose.PoseLandmark.NOSE.value],
    'shoulders': [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    'hips': [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value],
}

# Helferfunktion: Normalisierte Landmark -> Pixelkoordinate
def lm_to_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# Glättungs-Puffer für Landmark-Koordinaten (pro Landmark index)
class LandmarkSmoother:
    def __init__(self, n_landmarks, window=4):
        self.window = max(1, window)
        self.buffers = [deque(maxlen=self.window) for _ in range(n_landmarks)]

    def smooth(self, points):  # points: list of (x,y) or None
        smoothed = []
        for i, p in enumerate(points):
            if p is None:
                # kein Punkt erkannt -> leere Buffer nicht ändern, gib None zurück
                smoothed.append(None)
                continue
            self.buffers[i].append(p)
            # Mittelwert
            sx = sum(a[0] for a in self.buffers[i]) / len(self.buffers[i])
            sy = sum(a[1] for a in self.buffers[i]) / len(self.buffers[i])
            smoothed.append((int(sx), int(sy)))
        return smoothed

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    # MediaPipe Pose initialisieren
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    prev_time = time.time()
    smoother = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fehler: Frame konnte nicht gelesen werden.")
                break

            h, w = frame.shape[:2]
            # Flip horizontal, damit es wie ein Spiegel wirkt (optional)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            # Landmarken abfragen
            points = []
            if results.pose_landmarks:
                # initialisiere smoother wenn noch nicht vorhanden
                if smoother is None:
                    n_landmarks = len(results.pose_landmarks.landmark)
                    smoother = LandmarkSmoother(n_landmarks, window=SMOOTHING_WINDOW)

                for lm in results.pose_landmarks.landmark:
                    # nur sichtbare Landmarken verwenden (Sichtbarkeit > 0.2)
                    if hasattr(lm, 'visibility') and lm.visibility < 0.2:
                        points.append(None)
                    else:
                        points.append(lm_to_point(lm, w, h))
            else:
                # keine Landmarken erkannt
                points = [None] * 33  # mediapipe pose hat 33 Landmarks

            # glätten
            if smoother:
                points = smoother.smooth(points)

            # --- Zeichnen des Strichmännchens ---
            overlay = frame.copy()

            # Linien für Verbindungen
            for a, b in POSE_CONNECTIONS:
                pa = points[a] if a < len(points) else None
                pb = points[b] if b < len(points) else None
                if pa is None or pb is None:
                    continue
                # Linienstärke abhängig von Distanz (kleinerer Wert = dünner)
                length = ((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2) ** 0.5
                thickness = max(2, int(length / 100))
                cv2.line(overlay, pa, pb, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

            # Gelenk-Kreise
            for group in JOINTS_TO_DRAW.values():
                for idx in group:
                    if idx < len(points) and points[idx] is not None:
                        x, y = points[idx]
                        cv2.circle(overlay, (x, y), 6, (0, 120, 255), -1, lineType=cv2.LINE_AA)

            # Leicht transparentes Overlay (damit Kamera noch sichtbar bleibt)
            alpha = 0.9
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Optional: FPS
            if SHOW_FPS:
                now = time.time()
                fps = 1.0 / (now - prev_time) if (now - prev_time) > 1e-6 else 0.0
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # Anzeige
            cv2.imshow("Pose Stickman - q zum Beenden", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()
