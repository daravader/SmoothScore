import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import zipfile
import io
import os
import webbrowser

# URL to the Forest theme GitHub repository zip file
FOREST_THEME_REPO_ZIP = "https://github.com/rdbende/Forest-ttk-theme/archive/refs/heads/master.zip"
FOREST_THEME_DIR = "Forest-ttk-theme-master"

# ------------------------------------------------------------------
# Theme helper
# ------------------------------------------------------------------

def setup_forest_theme():
    """Download & extract Forest‑ttk theme the first time the app runs."""
    if not os.path.exists(FOREST_THEME_DIR):
        print("Downloading Forest theme …")
        try:
            response = requests.get(FOREST_THEME_REPO_ZIP, timeout=10)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall()
            print("Theme ready ✔")
        except requests.exceptions.RequestException as e:
            print(f"Theme download failed: {e}")
            return False
    return True

# ------------------------------------------------------------------
# MediaPipe initialisation
# ------------------------------------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------

camera_running = False
camera_source = 0  # 0 = default/laptop cam
current_pose = "Unknown"
current_gestures = []
STOP_FIGHT_POSES = {"Arms Extended", "T-pose"}

# ------------------------------------------------------------------
# Hand‑gesture utilities
# ------------------------------------------------------------------

def count_extended_fingers(hand_landmarks):
    """Return the number of raised fingers (index→pinky)."""
    finger_tips = [8, 12, 16, 20]
    return sum(1 for tip in finger_tips
               if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)


def classify_hand_gesture(hand_landmarks):
    """Map finger‑count or classic shapes → label/string."""
    fingers_up = count_extended_fingers(hand_landmarks)
    if fingers_up in (2, 3, 4):
        return f"{fingers_up} Points"

    # fallback mini‑set of gestures
    FINGER_TIPS = [4, 8, 12, 16, 20]
    states = [int(hand_landmarks.landmark[t].y < hand_landmarks.landmark[t-2].y)
              for t in FINGER_TIPS]
    if states == [1, 0, 0, 0, 0]:
        return "Thumb Up"
    if states == [1, 1, 1, 1, 1]:
        return "All Fingers Extended"
    return "Unknown"

# ------------------------------------------------------------------
# Pose utilities
# ------------------------------------------------------------------

def classify_pose(lm):
    ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lw, rw = lm[mp_pose.PoseLandmark.LEFT_WRIST], lm[mp_pose.PoseLandmark.RIGHT_WRIST]

    if abs(ls.y - rs.y) < 0.1:  # shoulders roughly level
        # wrists roughly level w/ shoulders → classic T
        if abs(lw.y - ls.y) < 0.1 and abs(rw.y - rs.y) < 0.1:
            return "T-pose"
        # wrists far left/right of shoulders → arms straight out
        if lw.x < ls.x and rw.x > rs.x:
            return "Arms Extended"
        return "Standing Upright"
    return "Unknown"

# ------------------------------------------------------------------
# Camera worker
# ------------------------------------------------------------------

def run_camera():
    global camera_running, current_pose, current_gestures

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose ---------------------------------------------------------
        pose_res = pose.process(rgb)
        if pose_res.pose_landmarks:
            current_pose = classify_pose(pose_res.pose_landmarks.landmark)
            mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            current_pose = "Unknown"

        # Hands --------------------------------------------------------
        hand_res = hands.process(rgb)
        current_gestures = []
        total_pts = 0

        if hand_res.multi_hand_landmarks:
            for hlm in hand_res.multi_hand_landmarks:
                gesture = classify_hand_gesture(hlm)
                current_gestures.append(gesture)
                if "Points" in gesture:
                    total_pts += int(gesture.split()[0])
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # -------------------- overlays --------------------------------
        cv2.putText(frame, f"Pose: {current_pose}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for idx, g in enumerate(current_gestures):
            cv2.putText(frame, f"Hand {idx+1}: {g}", (10, 80 + 40*idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Score Signalled: {total_pts}", (10, 80 + 40*len(current_gestures) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        if current_pose in STOP_FIGHT_POSES:
            h, w, _ = frame.shape
            cv2.putText(frame, "STOP FIGHT", (int(w*0.15), int(h*0.55)),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 5)

        cv2.imshow("BJJ Scoring Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------------------------
# GUI helpers
# ------------------------------------------------------------------

def start_camera():
    global camera_running
    if not camera_running:
        camera_running = True
        threading.Thread(target=run_camera, daemon=True).start()


def stop_camera():
    global camera_running
    camera_running = False


def set_camera_source(ip_digits):
    global camera_source
    try:
        o = ip_digits.split('.')
        if len(o) == 2 and all(i.isdigit() and 0 <= int(i) <= 255 for i in o):
            camera_source = f"http://192.168.{ip_digits}:8080/video"
            messagebox.showinfo("Camera", f"Switched to IP cam {camera_source}")
        else:
            raise ValueError
    except ValueError:
        messagebox.showerror("Input", "Enter last two octets like 0.212")


def reset_to_device_camera():
    global camera_source
    camera_source = 0
    messagebox.showinfo("Camera", "Back to built‑in cam")


def show_about():
    msg = (
        "Real‑time BJJ vision demo\n\n"
        "∙ Raise 2/3/4 fingers → 2/3/4 points.\n"
        "∙ Extend both arms sideways → STOP FIGHT signal."
    )
    messagebox.showinfo("About", msg)


def open_github():
    webbrowser.open("https://github.com/BorisJIordanov/BJJ")

# ------------------------------------------------------------------
# Build GUI
# ------------------------------------------------------------------

root = tk.Tk()
root.title("BJJ Vision Scoring Demo")
root.geometry("800x600")

if setup_forest_theme():
    try:
        root.tk.call("source", os.path.join(FOREST_THEME_DIR, "forest-dark.tcl"))
        ttk.Style().theme_use("forest-dark")
    except Exception as e:
        print("Theme init error:", e)

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

main_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="Live Demo")

header = "Raise 2/3/4 fingers → points | Extend arms → STOP FIGHT"

ttk.Label(main_tab, text=header, font=("Calibri", 14, "bold")).pack(pady=15)

ttk.Button(main_tab, text="Start Camera", command=start_camera).pack(pady=6)

ttk.Button(main_tab, text="Stop Camera (q)", command=stop_camera).pack(pady=6)

ip_frame = ttk.Frame(main_tab)
ip_frame.pack(pady=10)

ttk.Label(ip_frame, text="IP cam last two octets:").pack(side="left")
entry = ttk.Entry(ip_frame, width=10)
entry.pack(side="left", padx=4)

ttk.Button(ip_frame, text="Set", command=lambda: set_camera_source(entry.get())).pack(side="left")

ttk.Button(main_tab, text="Use Device Cam", command=reset_to_device_camera).pack(pady=6)

about_tab = ttk.Frame(notebook)
notebook.add(about_tab, text="About")

ttk.Label(about_tab, text="Vision‑based BJJ scoring and STOP signal", wraplength=700).pack(pady=20)

contact_tab = ttk.Frame(notebook)
notebook.add(contact_tab, text="Contact")

ttk.Button(contact_tab, text="GitHub Repository", command=open_github).pack(pady=20)

# Exit

ttk.Button(root, text="Exit", command=root.quit).pack(side="bottom", pady=15)

root.mainloop()
