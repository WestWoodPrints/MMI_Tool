import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Live Video Test")
        self.geometry("900x600")

        # Video Label
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(padx=20, pady=20, expand=True)

        # OpenCV Kamera
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Kamera konnte nicht geöffnet werden!")
            return

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # wichtig!

        self.after(10, self.update_frame)

if __name__ == "__main__":
    app = App()
    app.mainloop()
