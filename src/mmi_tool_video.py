import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from customtkinter import CTkImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time
import sys
from tkinter import filedialog
import csv

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

# ----------------------------------------------------------
# HELFER
# ----------------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def detect_local_extrema(data):
    minima = []
    maxima = []
    for i in range(1, len(data) - 1):
        if data[i] < data[i - 1] and data[i] < data[i + 1]:
            minima.append(i)
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            maxima.append(i)
    return minima, maxima

# ----------------------------------------------------------
# GUI
# ----------------------------------------------------------
class KneeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.title("MMI-Tool Kniewinkel Analyser")
        self.geometry("1400x1200")

        # Webcam Capture
        self.current_cam_index = 0
        self.cam_cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_DSHOW)
        if not self.cam_cap.isOpened():
            raise RuntimeError("Kamera konnte nicht geöffnet werden.")

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # Recording / Video
        self.recording = False
        self.angle_history = []

        self.video_path = None
        self.video_cap = None
        self.video_playing = False

        # Auto Recording
        self.auto_recording = False
        self.auto_duration = 10
        self.auto_start_time = None

        self.update_job = None

        # Layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # -------------------------
        # Video Label
        # -------------------------
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # -------------------------
        # Plot + Controls
        # -------------------------
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nse")
        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_title("Knie-Winkel Verlauf")
        self.ax.set_xlabel("Frames")
        self.ax.set_ylabel("Winkel (Grad)")
        self.line, = self.ax.plot([], [], label="Winkel")
        self.min_line, = self.ax.plot([], [], "go", label="Minima")
        self.max_line, = self.ax.plot([], [], "ro", label="Maxima")
        self.ax.legend()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=(10,0), sticky="nsew")

        # Info Frame
        info_frame = ctk.CTkFrame(right_frame)
        info_frame.grid(row=1, column=0, padx=10, pady=10, sticky="new")

        self.angle_label = ctk.CTkLabel(info_frame, text="Aktueller Winkel: --°", font=("Arial", 16))
        self.angle_label.pack(pady=10)
        self.min_label = ctk.CTkLabel(info_frame, text="Ø Minima: --°", font=("Arial", 15))
        self.min_label.pack()
        self.max_label = ctk.CTkLabel(info_frame, text="Ø Maxima: --°", font=("Arial", 15))
        self.max_label.pack()

        # Buttons
        btn_frame = ctk.CTkFrame(info_frame)
        btn_frame.pack(padx=10, pady=(10,10))
        self.start_btn = ctk.CTkButton(btn_frame, text="Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=10, pady=10)
        self.stop_btn = ctk.CTkButton(btn_frame, text="Stop", command=self.stop_recording)
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10)
        self.auto_btn = ctk.CTkButton(btn_frame, text=f"Auto Record ({self.auto_duration}s)",
                                      fg_color="red", hover_color="dark red", command=self.start_auto_recording)
        self.auto_btn.grid(row=0, column=2, padx=10, pady=10)
        self.clear_btn = ctk.CTkButton(btn_frame, text="Plot leeren", command=self.clear_plot)
        self.clear_btn.grid(row=0, column=3, padx=10, pady=10)

        # -------------------------
        # Video Upload / Start-Stop
        # -------------------------
        #self.file_label = ctk.CTkLabel(btn_frame, text="Keine Datei ausgewählt")
        #self.file_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.upload_btn = ctk.CTkButton(
            btn_frame,
            text="Video auswählen",
            command=self.select_video
        )
        self.upload_btn.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.video_start_btn = ctk.CTkButton(
            btn_frame,
            text="Video Start/Stop",
            command=self.toggle_video
        )
        self.video_start_btn.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        self.csv_btn = ctk.CTkButton(
            btn_frame,
            text="CSV Export",fg_color="red", hover_color="dark red",
            command=self.export_csv
        )
        self.csv_btn.grid(row=1, column=2, padx=10, pady=10, sticky="e")

        self.cam_label = ctk.CTkLabel(btn_frame, text="Kamera:")
        self.cam_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.cam_option = ctk.CTkOptionMenu(
            btn_frame,
            values=[str(i) for i in range(4)],
            command=self.change_camera
        )
        self.cam_option.set(str(self.current_cam_index))
        self.cam_option.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.duration_label = ctk.CTkLabel(btn_frame, text="Auto-Rec (s):")
        self.duration_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")

        self.duration_option = ctk.CTkOptionMenu(
            btn_frame,
            values=["5", "10", "20", "30", "60", "120"],
            command=self.update_auto_duration
        )
        self.duration_option.set(str(self.auto_duration))
        self.duration_option.grid(row=2, column=3, padx=10, pady=10, sticky="w")

        # Starte Frame Loop
        self.update_frame()

    # ------------------------------------------------------
    def select_video(self):
        filepath = filedialog.askopenfilename(
            title="Video auswählen",
            filetypes=[("Videos", "*.mp4"), ("Alle Dateien", "*.*")]
        )
        if not filepath:
            return
        self.video_path = filepath
        #self.file_label.configure(text=filepath.split("/")[-1])
        # Video Capture vorbereiten
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.video_playing = False

    def toggle_video(self):
        if self.video_cap is None:
            return
        self.video_playing = not self.video_playing
        if self.video_playing:
            #self.angle_history = []  # Neustart
            self.min_label.configure(text="Ø Minima: --°")
            self.max_label.configure(text="Ø Maxima: --°")
            self.angle_label.configure(text="Aktueller Winkel: --°")

    def change_camera(self, selection):
        idx = int(selection)
        if idx == self.current_cam_index:
            return
        
        if self.cam_cap is not None:
            self.cam_cap.release()
            
        self.current_cam_index = idx
        self.cam_cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_DSHOW)

    def update_auto_duration(self, selection):
        try:
            self.auto_duration = float(selection)
            self.auto_btn.configure(text=f"Auto Record ({self.auto_duration}s)")
        except ValueError:
            pass

    def export_csv(self):
        if not self.angle_history:
            return
            
        filepath = filedialog.asksaveasfilename(
            title="CSV Exportieren",
            defaultextension=".csv",
            filetypes=[("CSV Dateien", "*.csv"), ("Alle Dateien", "*.*")]
        )
        if filepath:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["Frame", "Winkel"])
                for i, angle in enumerate(self.angle_history):
                    writer.writerow([i, int(round(angle))])

    # ------------------------------------------------------
    def update_frame(self):
        if not self.winfo_exists():
            return

        # Entscheide welche Quelle
        cap = self.video_cap if (self.video_cap is not None and self.video_playing) else self.cam_cap

        ret, frame = cap.read()
        if not ret:
            if cap == self.video_cap:
                # Video Ende
                self.video_playing = False
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)  # schwarzes Bild
        h, w = frame.shape[:2]

        # Pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # Countdown Overlay
        if self.auto_recording and self.auto_start_time:
            elapsed = time.time() - self.auto_start_time
            remaining = max(0, int(self.auto_duration - elapsed))
            cv2.putText(frame, f"{remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)
            if remaining == 0:
                self.stop_recording()
                self.auto_recording = False
                self.auto_start_time = None
                self.set_buttons_state(False)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            hip = (int(lm[LEFT_HIP].x * w), int(lm[LEFT_HIP].y * h))
            knee = (int(lm[LEFT_KNEE].x * w), int(lm[LEFT_KNEE].y * h))
            ankle = (int(lm[LEFT_ANKLE].x * w), int(lm[LEFT_ANKLE].y * h))

            angle = calculate_angle(hip, knee, ankle)
            try:
                self.angle_label.configure(text=f"Aktueller Winkel: {angle:.1f}°")
            except Exception:
                pass

            if self.recording or self.video_playing:
                self.angle_history.append(angle)
                self.update_plot()

            # Linien einzeichnen
            cv2.line(frame, hip, knee, (0, 255, 0), 3)
            cv2.line(frame, knee, ankle, (0, 255, 0), 3)
            cv2.circle(frame, knee, 6, (0, 128, 255), -1)

        # Convert to CTkImage
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            ctk_img = CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img
        except Exception:
            pass

        # Schedule next frame
        try:
            if self.winfo_exists():
                self.update_job = self.after(10, self.update_frame)
        except Exception:
            self.update_job = None

    # ------------------------------------------------------
    def update_plot(self):
        if len(self.angle_history) < 3:
            return

        y = np.array(self.angle_history)
        x = np.arange(len(y))

        minima, maxima = detect_local_extrema(y)

        self.line.set_data(x, y)
        self.min_line.set_data(minima, y[minima] if len(minima) > 0 else [])
        self.max_line.set_data(maxima, y[maxima] if len(maxima) > 0 else [])

        try:
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------
    def start_recording(self):
        self.angle_history = []
        self.recording = True
        self.min_label.configure(text="Ø Minima: --°")
        self.max_label.configure(text="Ø Maxima: --°")

    def stop_recording(self):
        self.recording = False
        if len(self.angle_history) < 60:
            return

        y = np.array(self.angle_history[30:-30])
        minima_idx, maxima_idx = detect_local_extrema(y)

        lower_quartile = np.percentile(y, 10)
        filtered_min = [y[i] for i in minima_idx if y[i] <= lower_quartile]
        mean_min = np.mean(filtered_min) if filtered_min else np.nan

        upper_quartile = np.percentile(y, 90)
        filtered_max = [y[i] for i in maxima_idx if y[i] >= upper_quartile]
        mean_max = np.mean(filtered_max) if filtered_max else np.nan

        self.min_label.configure(
            text=f"Ø Minima: {mean_min:.1f}°" if not np.isnan(mean_min) else "Ø Minima: --°"
        )
        self.max_label.configure(
            text=f"Ø Maxima: {mean_max:.1f}°" if not np.isnan(mean_max) else "Ø Maxima: --°"
        )

    def start_auto_recording(self):
        if self.auto_recording:
            return

        try:
            val = float(self.duration_option.get())
            if val > 0:
                self.auto_duration = val
                self.auto_btn.configure(text=f"Auto Record ({self.auto_duration}s)")
        except ValueError:
            pass

        self.start_recording()
        self.auto_recording = True
        self.auto_start_time = time.time()
        self.set_buttons_state(True)

    def clear_plot(self):
        if self.recording or self.auto_recording:
            return
        self.angle_history = []
        self.line.set_data([], [])
        self.min_line.set_data([], [])
        self.max_line.set_data([], [])
        try:
            self.canvas.draw_idle()
        except Exception:
            pass
        self.min_label.configure(text="Ø Minima: --°")
        self.max_label.configure(text="Ø Maxima: --°")
        self.angle_label.configure(text="Aktueller Winkel: --°")

    def set_buttons_state(self, auto_recording):
        if auto_recording:
            self.start_btn.configure(state="disabled", fg_color="#303030")
            self.stop_btn.configure(state="disabled", fg_color="#303030")
            self.clear_btn.configure(state="disabled", fg_color="#303030")
            self.auto_btn.configure(fg_color="red", state="normal")
        else:
            self.start_btn.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self.stop_btn.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self.clear_btn.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            self.auto_btn.configure(fg_color="red")

    # ------------------------------------------------------
    def on_closing(self):
        self.recording = False
        self.auto_recording = False
        self.video_playing = False

        if hasattr(self, "update_job") and self.update_job is not None:
            try:
                self.after_cancel(self.update_job)
            except Exception:
                pass
            self.update_job = None

        try:
            if hasattr(self, "cam_cap") and self.cam_cap.isOpened():
                self.cam_cap.release()
            if hasattr(self, "video_cap") and self.video_cap is not None:
                self.video_cap.release()
            if hasattr(self, "pose"):
                self.pose.close()
        except Exception:
            pass

        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass

        sys.exit(0)

# ------------------------------------------------------
if __name__ == "__main__":
    app = KneeApp()
    app.mainloop()
