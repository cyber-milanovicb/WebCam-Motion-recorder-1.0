
import cv2
import os
import time
import logging
import threading
import tkinter as tk

from datetime import datetime
from PIL import Image, ImageTk, ImageDraw
import pystray

# ==========================================
# SETTINGS
# ==========================================
CAMERA_INDEX = 0
MOTION_THRESHOLD = 1500
STOP_AFTER_SECONDS = 5
FPS = 20.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "recordings")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_FOLDER, "motion_recorder.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
cap = cv2.VideoCapture("192.168.1.100/8080/recordings")
stop_event = threading.Event()
lock = threading.Lock()

shared = {
    "frame": None,
    "active": False,
    "recording": False,
    "camera_ok": False,
}

# ==========================================
# CAMERA AND MOTION DETECTION THREAD
# ==========================================
def camera_worker():
    camera = None
    writer = None
    recording = False
    previous_gray = None
    last_motion_time = 0

    def open_camera():
        cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return cam

    try:
        camera = open_camera()

        while not stop_event.is_set():

            if camera is None or not camera.isOpened():
                if camera is not None:
                    camera.release()

                logging.info("Attempting to reconnect to webcam.")
                time.sleep(2)

                if stop_event.is_set():
                    break

                camera = open_camera()
                previous_gray = None
                continue

            success, frame = camera.read()

            # Webcam can disconnect or stop responding after sleep.
            if not success:
                logging.warning("Camera read failed; reconnecting.")

                if writer is not None:
                    writer.release()
                    writer = None

                recording = False
                previous_gray = None

                with lock:
                    shared["camera_ok"] = False
                    shared["active"] = False
                    shared["recording"] = False

                camera.release()
                camera = None
                time.sleep(2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            motion_area = 0

            if previous_gray is not None:
                difference = cv2.absdiff(previous_gray, gray)

                _, mask = cv2.threshold(
                    difference, 25, 255, cv2.THRESH_BINARY
                )

                mask = cv2.dilate(mask, None, iterations=2)

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = cv2.contourArea(contour)

                    if area > 500:
                        motion_area += area

                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

            previous_gray = gray.copy()
            now = time.time()
            motion_detected = motion_area > MOTION_THRESHOLD

            if motion_detected:
                last_motion_time = now

                if not recording:
                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )

                    filename = os.path.join(
                        OUTPUT_FOLDER,
                        f"motion_{timestamp}.avi"
                    )

                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")

                    writer = cv2.VideoWriter(
                        filename,
                        fourcc,
                        FPS,
                        (width, height)
                    )

                    if writer.isOpened():
                        recording = True
                        logging.info("Recording started: %s", filename)
                    else:
                        logging.error(
                            "Could not create video: %s", filename
                        )
                        writer.release()
                        writer = None

            if recording and writer is not None:
                writer.write(frame)

                if now - last_motion_time >= STOP_AFTER_SECONDS:
                    writer.release()
                    writer = None
                    recording = False
                    logging.info("Recording stopped and saved.")

            # Keep the camera window active during recording,
            # including the final seconds after movement.
            with lock:
                shared["frame"] = frame.copy()
                shared["camera_ok"] = True
                shared["recording"] = recording
                shared["active"] = motion_detected or recording

    except Exception:
        logging.exception("Camera worker failed.")

    finally:
        if writer is not None:
            writer.release()

        if camera is not None:
            camera.release()

        with lock:
            shared["camera_ok"] = False
            shared["active"] = False
            shared["recording"] = False

        logging.info("Camera worker stopped.")


# ==========================================
# SYSTEM TRAY ICON
# ==========================================
def create_tray_image():
    image = Image.new("RGB", (64, 64), (40, 40, 40))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (8, 18, 48, 46),
        radius=5,
        fill=(230, 230, 230)
    )
    draw.polygon(
        [(48, 24), (59, 17), (59, 47), (48, 40)],
        fill=(80, 200, 120)
    )
    draw.ellipse((19, 22, 38, 41), fill=(40, 40, 40))
    draw.ellipse((24, 27, 33, 36), fill=(80, 200, 120))

    return image


# ==========================================
# WINDOW AND USER INTERFACE
# ==========================================
def main():
    root = tk.Tk()
    root.title("Webcam Motion Recorder")
    root.geometry("680x540")
    root.resizable(False, False)

    status_label = tk.Label(
        root,
        text="Waiting for movement...",
        font=("Segoe UI", 12, "bold")
    )
    status_label.pack(pady=8)

    video_label = tk.Label(root, text="Camera is waiting for motion")
    video_label.pack(padx=10, pady=5)

    hint_label = tk.Label(
        root,
        text="The window hides automatically when the scene is still.",
        font=("Segoe UI", 9)
    )
    hint_label.pack(pady=5)

    # Start hidden; tray icon remains available.
    root.withdraw()

    def hide_window():
        root.withdraw()

    root.protocol("WM_DELETE_WINDOW", hide_window)

    def show_window(icon=None, item=None):
        root.after(0, root.deiconify)
        root.after(50, root.lift)

    def exit_app(icon=None, item=None):
        stop_event.set()

        if icon is not None:
            icon.stop()

        root.after(0, root.destroy)

    tray = pystray.Icon(
        "MotionRecorder",
        create_tray_image(),
        "Webcam Motion Recorder",
        menu=pystray.Menu(
            pystray.MenuItem("Show window", show_window),
            pystray.MenuItem("Exit", exit_app)
        )
    )

    threading.Thread(
        target=tray.run,
        daemon=True
    ).start()

    threading.Thread(
        target=camera_worker,
        daemon=True
    ).start()

    last_image = None

    def update_window():
        nonlocal last_image

        if stop_event.is_set():
            return

        with lock:
            frame = (
                shared["frame"].copy()
                if shared["frame"] is not None
                else None
            )
            active = shared["active"]
            recording = shared["recording"]
            camera_ok = shared["camera_ok"]

        if active and frame is not None:
            # Show the window when motion is detected.
            if not root.winfo_viewable():
                root.deiconify()
                root.lift()

            if recording:
                status_label.config(
                    text="● RECORDING",
                    fg="red"
                )
            else:
                status_label.config(
                    text="Motion detected",
                    fg="darkorange"
                )

            # Convert OpenCV BGR image to Tkinter RGB image.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=image)

            video_label.config(image=photo, text="")
            video_label.image = photo
            last_image = photo

        else:
            # Hide automatically when motion/recording ends.
            if root.winfo_viewable():
                root.withdraw()

            if not camera_ok:
                status_label.config(
                    text="Waiting for camera...",
                    fg="gray"
                )
            else:
                status_label.config(
                    text="Waiting for movement...",
                    fg="green"
                )

        root.after(100, update_window)

    root.after(100, update_window)

    try:
        root.mainloop()
    finally:
        stop_event.set()
        tray.stop()


if __name__ == "__main__":
    main()
