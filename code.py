import face_recognition
import cv2
import numpy as np
import csv
import os
import serial
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from queue import Queue
import sys

#-----spoofing config------
spoof_config = {
    "enabled": True,
    "check_duration": 3, 
    "frame_count": 15,
    "threshold": 0.40,
}

#eye blink detection settings
EYE_BLINK_THRESHOLD = 0.25
BLINK_REQUIREMENT = 2   

# TIMEOUT SETTINGS
FACE_DETECTION_TIMEOUT = 30 
RFID_TIMEOUT = 10 


#------------FACE LANDMARKS FOR ANTI-SPOOFING-------------
# For facial landmark detection
FACIAL_LANDMARKS_68_IDXS = {
    "mouth": (48, 68),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "nose": (27, 36),
    "jaw": (0, 17)
}

#---------ANTI-SPOOFING FUNCTIONS-----
class AntiSpoofingDetector:
    def __init__(self):
        self.eye_blink_counter = 0
        self.frame_buffer = []
        self.blink_detected = False
        self.last_blink_time = None
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio for blink detection"""
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blinks(self, landmarks):
        """Detect eye blinks using facial landmarks"""
        # Get eye landmarks
        left_eye = landmarks[FACIAL_LANDMARKS_68_IDXS["left_eye"][0]:FACIAL_LANDMARKS_68_IDXS["left_eye"][1]]
        right_eye = landmarks[FACIAL_LANDMARKS_68_IDXS["right_eye"][0]:FACIAL_LANDMARKS_68_IDXS["right_eye"][1]]
        
        # Calculate eye aspect ratios
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        
        # Average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0
        
        # Check for blink
        if ear < EYE_BLINK_THRESHOLD:
            if not self.blink_detected:
                self.eye_blink_counter += 1
                self.blink_detected = True
                self.last_blink_time = datetime.now()
                return True
        else:
            self.blink_detected = False
            
        return False
    
    def analyze_texture(self, face_region):
        """Analyze texture patterns for spoof detection"""
        if face_region.size == 0:
            return 1.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Calculate texture metrics
        # 1. Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 2. Local Binary Patterns (LBP) - simplified version
        lbp_value = self.simplified_lbp(gray)
        
        # 3. Color saturation analysis (photos often have different saturation)
        if len(face_region.shape) == 3:
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1])
        else:
            saturation = 0
        
        # Combine metrics (simplified - in practice you'd train a model)
        spoof_score = self.compute_spoof_score(sharpness, lbp_value, saturation)
        
        return spoof_score
    
    def simplified_lbp(self, gray_image):
        """Simplified Local Binary Pattern calculation"""
        height, width = gray_image.shape
        lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray_image[i, j]
                code = 0
                code |= (gray_image[i-1, j-1] > center) << 7
                code |= (gray_image[i-1, j] > center) << 6
                code |= (gray_image[i-1, j+1] > center) << 5
                code |= (gray_image[i, j+1] > center) << 4
                code |= (gray_image[i+1, j+1] > center) << 3
                code |= (gray_image[i+1, j] > center) << 2
                code |= (gray_image[i+1, j-1] > center) << 1
                code |= (gray_image[i, j-1] > center) << 0
                lbp_image[i-1, j-1] = code
        
        # Calculate histogram (simplified)
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        # Return entropy as texture measure
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return entropy
    
    def compute_spoof_score(self, sharpness, texture_entropy, saturation):
        """Compute combined spoof score"""
        # ADJUSTED NORMALIZATION VALUES
        sharpness_norm = min(sharpness / 300.0, 1.0)  # REDUCED from 500 to 300
        texture_norm = min(texture_entropy / 6.0, 1.0)  # REDUCED from 8.0 to 6.0
        saturation_norm = min(saturation / 150.0, 1.0)  # INCREASED from 100 to 150
        
        # ADJUSTED WEIGHTS - give more weight to motion/blinks
        spoof_score = (0.3 * sharpness_norm + 0.3 * texture_norm + 0.1 * saturation_norm)
        
        return spoof_score
    
    def check_motion(self, frame_buffer):
        """Check for natural micro-movements across frames"""
        if len(frame_buffer) < 2:
            return 0.3  # CHANGED from 0.5 to 0.3
            
        # Calculate optical flow between consecutive frames
        motion_scores = []
        
        for i in range(len(frame_buffer) - 1):
            prev_gray = cv2.cvtColor(frame_buffer[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame_buffer[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude of flow
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            
            motion_scores.append(avg_magnitude)
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        
        # If motion is very low, give benefit of doubt
        if avg_motion < 0.1:
            return 0.3  # Favor real face for very low motion
        
        # Normalize motion score
        motion_score = min(avg_motion / 2.0, 1.0)  # CHANGED from 3.0 to 2.0
        
        return motion_score
    
    def detect_spoof(self, frame, face_location, landmarks=None):
        """
        Main spoof detection function
        Returns: (is_spoof, confidence_score, detection_info)
        """
        if not spoof_config["enabled"]:
            return False, 0.0, "Spoof detection disabled"
        
        # Extract face region with padding
        top, right, bottom, left = face_location
        
        # Add 20-pixel padding to get better texture analysis
        padding = 20
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(frame.shape[1], right + padding)
        
        face_region = frame[top:bottom, left:right]
        
        # Check if face region is too small
        if face_region.size == 0 or face_region.shape[0] < 50 or face_region.shape[1] < 50:
            print("⚠️ Face region too small for proper analysis")
            return False, 0.3, "Small face region"
        
        # Add to frame buffer for motion analysis
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > spoof_config["frame_count"]:
            self.frame_buffer.pop(0)
        
        # Calculate various spoof indicators
        texture_score = self.analyze_texture(face_region)
        motion_score = self.check_motion(self.frame_buffer) if len(self.frame_buffer) > 1 else 0.3
        
        # Check for blinks if landmarks available
        blink_detected = False
        if landmarks is not None:
            blink_detected = self.detect_blinks(landmarks)
        
        # DEBUG OUTPUT
        print(f"\n🔍 SPOOF ANALYSIS:")
        print(f"   Texture: {texture_score:.3f}")
        print(f"   Motion: {motion_score:.3f}")
        print(f"   Blinks: {self.eye_blink_counter}/{BLINK_REQUIREMENT}")
        
        # Combined spoof score with better blink handling
        base_score = (0.6 * texture_score + 0.4 * (1.0 - motion_score))
        
        # STRONGER blink adjustment
        if self.eye_blink_counter >= BLINK_REQUIREMENT:
            base_score *= 0.4  # CHANGED from 0.7 to 0.4 (60% reduction instead of 30%)
            print(f"   ✓ Blink discount: -60%")
        elif self.eye_blink_counter > 0:
            base_score *= 0.7  # Moderate reduction if some blinks
            print(f"   ✓ Partial blink adjustment: -30%")
        else:
            base_score *= 1.2  # Penalty if no blinks
            print(f"   ✗ No blink penalty: +20%")
        
        spoof_score = min(base_score, 1.0)  # Cap at 1.0
        print(f"   Final score: {spoof_score:.3f} (Threshold: {spoof_config['threshold']})")
        
        # Decision
        is_spoof = spoof_score > spoof_config["threshold"]
        
        detection_info = {
            "texture_score": texture_score,
            "motion_score": motion_score,
            "blinks": self.eye_blink_counter,
            "spoof_score": spoof_score,
            "threshold": spoof_config["threshold"]
        }
        
        return is_spoof, spoof_score, detection_info
    
    def reset(self):
        """Reset detector state for new verification"""
        self.eye_blink_counter = 0
        self.frame_buffer = []
        self.blink_detected = False
        self.last_blink_time = None

# Initialize anti-spoofing detector
anti_spoof = AntiSpoofingDetector()

# ============================================================================
# RFID DATABASE
# ============================================================================
rfid_database = {
    "Med Hedi Abd": "11221122",
}

# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================
try:
    ser = serial.Serial(port='COM1', baudrate=9600, timeout=1)
    time.sleep(2)
    print("COM port connected")
except Exception as e:
    print(f"COM port error: {e}")
    ser = None

# ============================================================================
# FACE RECOGNITION SETUP
# ============================================================================
video_capture = cv2.VideoCapture(0)

known_face_encoding = []
known_face_names = []

face_data = {
    "medhedi.png": "Med Hedi Abd"
}

print("Loading faces...")
for image_file, name in face_data.items():
    try:
        image_path = f"images/{image_file}"
        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encoding.append(encoding)
                known_face_names.append(name)
                print(f"Loaded: {name}")
    except Exception as e:
        print(f"Error loading {image_file}: {e}")

if not known_face_encoding:
    print("ERROR: No faces loaded!")
    exit()

students = known_face_names.copy()

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
face_locations = []
face_encodings = []
face_names = []
process_frame = True
system_running = True  # Control variable for system state

# Timeout settings
timeout_minutes = 15
start_time = datetime.now()
timeout_time = start_time + timedelta(minutes=timeout_minutes)

# Anti-spoofing state
spoof_check_active = False
spoof_check_start_time = None
current_spoof_check_target = None
spoof_check_results = {}

# Background image
try:
    imgbackground = cv2.imread("background_image.jpg")
    if imgbackground is not None:
        bg_height, bg_width = imgbackground.shape[:2]
        cam_width, cam_height = 640, 480
        cam_x = (bg_width - cam_width) // 2
        cam_y = 120
        print("Background image loaded")
except Exception as e:
    imgbackground = np.full((700, 900, 3), [40, 40, 80], dtype=np.uint8)
    cam_x, cam_y, cam_width, cam_height = 130, 120, 640, 480
    print("Using default background")

# CSV file
current_date = datetime.now().strftime("%d-%m-%Y")
fichier = open(f'attendance_{current_date}.csv', 'w+', newline='', encoding='utf-8')
lnwriter = csv.writer(fichier)
lnwriter.writerow(["Name", "Status", "Time", "RFID_Verified", "Spoof_Checked"])

marked_students = set()
last_face_detection_time = {}
face_detection_cooldown = 30
rfid_timeout = 10
pending_verification = {}

# Thread-safe variables
rfid_input_queue = Queue()
rfid_input_expected = False
current_verification_target = None
rfid_window_active = False
unknown_face_cooldown = 5  # Cooldown for unknown face detection
last_unknown_detection = None

# ============================================================================
# RFID DENIED WINDOW CLASS
# ============================================================================
class RFIDDeniedWindow:
    def __init__(self, student_name):
        self.student_name = student_name
        self.window = None
        self.create_window()
    
    def create_window(self):
        """Create simple RFID denied window"""
        self.window = tk.Tk()
        self.window.title("RFID Verification Failed")
        self.window.geometry("400x200")
        self.window.configure(bg='#2c3e50')
        self.window.resizable(False, False)
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Center window on screen
        window_width = 400
        window_height = 200
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Warning icon
        warning_label = tk.Label(
            self.window,
            text="✗",
            font=("Arial", 36),
            fg="#e74c3c",
            bg="#2c3e50"
        )
        warning_label.pack(pady=20)
        
        # Message
        message_label = tk.Label(
            self.window,
            text="RFID Verification Failed",
            font=("Arial", 16, "bold"),
            fg="#e74c3c",
            bg="#2c3e50"
        )
        message_label.pack()
        
        # Auto close after 2 seconds
        self.window.after(2000, self.close_window)
    
    def close_window(self):
        """Close the RFID denied window"""
        if self.window:
            self.window.destroy()

# ============================================================================
# ACCESS GRANTED WINDOW CLASS
# ============================================================================
class AccessGrantedWindow:
    def __init__(self, student_name):
        self.student_name = student_name
        self.window = None
        self.create_window()
    
    def create_window(self):
        """Create simple access granted window"""
        self.window = tk.Tk()
        self.window.title("Access Granted")
        self.window.geometry("400x200")
        self.window.configure(bg='#2c3e50')
        self.window.resizable(False, False)
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Center window on screen
        window_width = 400
        window_height = 200
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Success icon
        success_label = tk.Label(
            self.window,
            text="✓",
            font=("Arial", 36),
            fg="#27ae60",
            bg="#2c3e50"
        )
        success_label.pack(pady=20)
        
        # Message
        message_label = tk.Label(
            self.window,
            text=f"Welcome {self.student_name}",
            font=("Arial", 16, "bold"),
            fg="#27ae60",
            bg="#2c3e50"
        )
        message_label.pack()
        
        # Auto close after 2 seconds
        self.window.after(2000, self.close_window)
    
    def close_window(self):
        """Close the access granted window"""
        if self.window:
            self.window.destroy()

# ============================================================================
# SPOOF DETECTION WINDOW CLASS
# ============================================================================
class SpoofCheckWindow:
    def __init__(self, student_name):
        self.student_name = student_name
        self.window = None
        self.message_label = None
        self.countdown_label = None
        self.countdown_value = spoof_config["check_duration"]
        self.countdown_active = True
        self.create_window()
    
    def create_window(self):
        """Create spoof check window"""
        self.window = tk.Tk()
        self.window.title("Anti-Spoofing Check")
        self.window.geometry("500x300")
        self.window.configure(bg='#2c3e50')
        self.window.resizable(False, False)
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Center window on screen
        window_width = 500
        window_height = 300
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Title
        title_label = tk.Label(
            self.window,
            text="ANTI-SPOOFING VERIFICATION",
            font=("Arial", 18, "bold"),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        title_label.pack(pady=20)
        
        # Student info
        student_label = tk.Label(
            self.window,
            text=f"Student: {self.student_name}",
            font=("Arial", 14),
            fg="#3498db",
            bg="#2c3e50"
        )
        student_label.pack(pady=10)
        
        # Instructions
        self.message_label = tk.Label(
            self.window,
            text="Please look at the camera and blink naturally...",
            font=("Arial", 12),
            fg="#f39c12",
            bg="#2c3e50"
        )
        self.message_label.pack(pady=10)
        
        # Countdown timer
        countdown_frame = tk.Frame(self.window, bg="#2c3e50")
        countdown_frame.pack(pady=10)
        
        self.countdown_label = tk.Label(
            countdown_frame,
            text=f"Checking: {self.countdown_value}s",
            font=("Arial", 12, "bold"),
            fg="#f39c12",
            bg="#2c3e50"
        )
        self.countdown_label.pack()
        
        # Start countdown
        self.start_countdown()
    
    def update_message(self, message, color="#f39c12"):
        """Update the instruction message"""
        if self.message_label:
            self.message_label.config(text=message, fg=color)
    
    def start_countdown(self):
        """Start countdown timer"""
        if self.countdown_value > 0 and self.countdown_active:
            self.countdown_label.config(text=f"Checking: {self.countdown_value}s")
            self.countdown_value -= 1
            self.window.after(1000, self.start_countdown)
        elif self.countdown_value == 0:
            self.countdown_label.config(text="ANALYSIS COMPLETE", fg="#27ae60")
            self.countdown_active = False
            self.window.after(1000, self.complete_check)
    
    def complete_check(self):
        """Complete the spoof check"""
        global spoof_check_active
        spoof_check_active = False
        if self.window:
            self.window.destroy()
    
    def force_close(self):
        """Force close the window"""
        global spoof_check_active
        self.countdown_active = False
        spoof_check_active = False
        if self.window:
            self.window.destroy()

# ============================================================================
# SPOOF DETECTED WINDOW CLASS
# ============================================================================
class SpoofDetectedWindow:
    def __init__(self, student_name, spoof_info):
        self.student_name = student_name
        self.spoof_info = spoof_info
        self.window = None
        self.create_window()
    
    def create_window(self):
        """Create spoof detected warning window"""
        self.window = tk.Tk()
        self.window.title("SPOOFING DETECTED!")
        self.window.geometry("500x350")
        self.window.configure(bg='#2c3e50')
        self.window.resizable(False, False)
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Center window on screen
        window_width = 500
        window_height = 350
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Warning icon
        warning_label = tk.Label(
            self.window,
            text="⚠️",
            font=("Arial", 36),
            fg="#e74c3c",
            bg="#2c3e50"
        )
        warning_label.pack(pady=20)
        
        # Main warning
        warning_text = tk.Label(
            self.window,
            text="SPOOFING ATTEMPT DETECTED!",
            font=("Arial", 16, "bold"),
            fg="#e74c3c",
            bg="#2c3e50"
        )
        warning_text.pack()
        
        # Details
        details_frame = tk.Frame(self.window, bg="#2c3e50")
        details_frame.pack(pady=20, padx=20)
        
        # Spoof score
        spoof_score = self.spoof_info.get('spoof_score', 0)
        threshold = self.spoof_info.get('threshold', spoof_config["threshold"])
        
        score_label = tk.Label(
            details_frame,
            text=f"Spoof Score: {spoof_score:.3f} (Threshold: {threshold})",
            font=("Arial", 11),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        score_label.grid(row=0, column=0, sticky="w", pady=5)
        
        # Texture score
        texture_score = self.spoof_info.get('texture_score', 0)
        texture_label = tk.Label(
            details_frame,
            text=f"Texture Analysis: {texture_score:.3f}",
            font=("Arial", 11),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        texture_label.grid(row=1, column=0, sticky="w", pady=5)
        
        # Motion score
        motion_score = self.spoof_info.get('motion_score', 0)
        motion_label = tk.Label(
            details_frame,
            text=f"Motion Detection: {motion_score:.3f}",
            font=("Arial", 11),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        motion_label.grid(row=2, column=0, sticky="w", pady=5)
        
        # Blink count
        blinks = self.spoof_info.get('blinks', 0)
        blink_label = tk.Label(
            details_frame,
            text=f"Blinks Detected: {blinks} (Required: {BLINK_REQUIREMENT})",
            font=("Arial", 11),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        blink_label.grid(row=3, column=0, sticky="w", pady=5)
        
        # Auto close after 5 seconds
        self.window.after(5000, self.close_window)
    
    def close_window(self):
        """Close the spoof detected window"""
        if self.window:
            self.window.destroy()

# ============================================================================
# RFID WINDOW CLASS
# ============================================================================
class RFIDWindow:
    def __init__(self, student_name, spoof_verified=True):
        self.student_name = student_name
        self.spoof_verified = spoof_verified
        self.window = None
        self.entry = None
        self.countdown_label = None
        self.countdown_value = RFID_TIMEOUT
        self.countdown_active = True
        self.create_window()
    
    def create_window(self):
        """Create RFID input window"""
        self.window = tk.Tk()
        self.window.title("RFID Verification")
        self.window.geometry("500x400")
        self.window.configure(bg='#2c3e50')
        self.window.resizable(False, False)
        
        # Make window stay on top
        self.window.attributes('-topmost', True)
        
        # Center window on screen
        window_width = 500
        window_height = 400
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Title with anti-spoofing indicator
        title_text = "RFID CARD SCANNER"
        if self.spoof_verified:
            title_text += " ✓"
        
        title_label = tk.Label(
            self.window,
            text=title_text,
            font=("Arial", 18, "bold"),
            fg="#27ae60" if self.spoof_verified else "#e74c3c",
            bg="#2c3e50"
        )
        title_label.pack(pady=15)
        
        # Student info
        student_label = tk.Label(
            self.window,
            text=f"Student: {self.student_name}",
            font=("Arial", 14),
            fg="#3498db",
            bg="#2c3e50"
        )
        student_label.pack(pady=10)
        
        # Anti-spoofing status
        spoof_status = tk.Label(
            self.window,
            text="✓ Anti-Spoofing: PASSED" if self.spoof_verified else "✗ Anti-Spoofing: FAILED",
            font=("Arial", 12, "bold"),
            fg="#27ae60" if self.spoof_verified else "#e74c3c",
            bg="#2c3e50"
        )
        spoof_status.pack(pady=5)
        
        # RFID input frame
        input_frame = tk.Frame(self.window, bg="#2c3e50")
        input_frame.pack(pady=20)
        
        rfid_label = tk.Label(
            input_frame,
            text="Enter RFID (8 digits):",
            font=("Arial", 12),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        rfid_label.grid(row=0, column=0, padx=(0, 10))
        
        self.entry = tk.Entry(
            input_frame,
            font=("Arial", 14, "bold"),
            width=20,
            justify="center",
            show="*",
            bg="#34495e",
            fg="#ecf0f1",
            insertbackground="#ecf0f1",
            relief="flat",
            bd=2
        )
        self.entry.grid(row=0, column=1)
        self.entry.focus_set()
        
        # Bind Enter key
        self.entry.bind('<Return>', self.submit_rfid)
        
        # Bind digit keys to auto-fill
        for i in range(10):
            self.entry.bind(str(i), self.on_digit_press)
        
        # Countdown timer
        countdown_frame = tk.Frame(self.window, bg="#2c3e50")
        countdown_frame.pack(pady=10)
        
        self.countdown_label = tk.Label(
            countdown_frame,
            text=f"Time left: {self.countdown_value}s",
            font=("Arial", 12, "bold"),
            fg="#e74c3c",
            bg="#2c3e50"
        )
        self.countdown_label.pack()
        
        # Button frame
        button_frame = tk.Frame(self.window, bg="#2c3e50")
        button_frame.pack(pady=20)
        
        submit_btn = tk.Button(
            button_frame,
            text="SUBMIT",
            command=self.submit_rfid,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=30,
            pady=10,
            relief="raised",
            bd=0,
            cursor="hand2"
        )
        submit_btn.grid(row=0, column=0, padx=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="CANCEL",
            command=self.cancel_verification,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=30,
            pady=10,
            relief="raised",
            bd=0,
            cursor="hand2"
        )
        cancel_btn.grid(row=0, column=1, padx=10)
        
        # Instructions
        instructions = tk.Label(
            self.window,
            text="Type 8 digits or scan RFID card",
            font=("Arial", 10),
            fg="#bdc3c7",
            bg="#2c3e50"
        )
        instructions.pack(pady=10)
        
        # Start countdown
        self.start_countdown()
    
    def on_digit_press(self, event):
        """Auto-focus next position after digit entry"""
        if self.entry and len(self.entry.get()) >= 8:
            self.entry.delete(8, tk.END)
            return "break"
    
    def start_countdown(self):
        """Start countdown timer"""
        if self.countdown_value > 0 and self.countdown_active:
            self.countdown_label.config(text=f"Time left: {self.countdown_value}s")
            self.countdown_value -= 1
            self.window.after(1000, self.start_countdown)
        elif self.countdown_value == 0:
            self.countdown_label.config(text="TIME OUT!", fg="#ff0000")
            self.countdown_active = False
            self.window.after(1000, self.timeout_close)
    
    def timeout_close(self):
        """Close window on timeout"""
        if self.window:
            rfid_input_queue.put(("TIMEOUT", self.student_name))
            self.window.destroy()
    
    def submit_rfid(self, event=None):
        """Submit RFID for verification"""
        rfid_value = self.entry.get().strip()
        
        if len(rfid_value) == 8 and rfid_value.isdigit():
            self.countdown_active = False
            rfid_input_queue.put((rfid_value, self.student_name, self.spoof_verified))
            if self.window:
                self.window.destroy()
        else:
            messagebox.showerror("Invalid RFID", "RFID must be 8 digits!")
            self.entry.delete(0, tk.END)
            self.entry.focus_set()
    
    def cancel_verification(self):
        """Cancel RFID verification"""
        self.countdown_active = False
        rfid_input_queue.put(("CANCELLED", self.student_name))
        if self.window:
            self.window.destroy()

# ============================================================================
# COMMUNICATION FUNCTIONS
# ============================================================================
def send_to_proteus(student_id, status="ACCESS"):
    """
    Send student_id to Proteus with status
    Format: "ID:STATUS" where STATUS can be:
    - SCANNING: when face is detected and spoof check passes
    - DENIED: when spoofing is detected OR RFID is wrong OR unknown face
    - Hi [Name]: when face + RFID are verified
    """
    if not ser:
        return False
        
    try:
        # Map student names to IDs
        student_id_map = {
            "Med Hedi Abd": '1'
        }
        
        mapped_id = student_id_map.get(student_id, '0')  # '0' for unknown
        
        # For Proteus compatibility, convert statuses:
        if status == "SPOOFING":
            # Send DENIED for spoofing (Proteus will show "SPOOF DETECTED!")
            status_to_send = "DENIED"
        elif status == "RFID_WRONG":
            # Send DENIED for wrong RFID (Proteus will show "WRONG RFID CARD")
            status_to_send = "DENIED"
        elif status.startswith("Hi "):
            # Keep "Hi [Name]" for successful verification
            status_to_send = f"Hi {student_id}"
        else:
            # For other cases (SCANNING, DENIED), send as is
            status_to_send = status
        
        # Send data to Proteus
        data = f"{mapped_id}:{status_to_send}\n"
        ser.write(data.encode('utf-8'))
        ser.flush()
        print(f"Sent to Proteus: {data.strip()}")
        return True
    except Exception as e:
        print(f"Serial error: {e}")
        return False
    
def verify_rfid(name, rfid_input):
    """Verify RFID against database"""
    if name in rfid_database:
        return rfid_input.strip() == rfid_database[name]
    return False

def perform_spoof_check(name, face_location, frame):
    """
    Perform anti-spoofing check for a detected face
    Returns: (is_real, spoof_info)
    """
    global anti_spoof, spoof_check_results
    
    print(f"\n🔍 Starting anti-spoofing check for {name}...")
    
    # Reset anti-spoofing detector
    anti_spoof.reset()
    
    # Get facial landmarks for this face
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Get face landmarks
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    
    landmarks = None
    if face_landmarks_list:
        # Convert landmarks to numpy array format
        landmarks_array = []
        for facial_feature in face_landmarks_list[0].values():
            for point in facial_feature:
                landmarks_array.append([point[0], point[1]])
        landmarks = np.array(landmarks_array) * 4  # Scale back up
        print(f"✓ Landmarks detected: {len(landmarks)} points")
    else:
        print("✗ No landmarks detected")
    
    # Perform spoof check WITH landmarks
    is_spoof, spoof_score, spoof_info = anti_spoof.detect_spoof(frame, face_location, landmarks)
    
    # Store results
    spoof_check_results[name] = {
        'is_spoof': is_spoof,
        'spoof_score': spoof_score,
        'info': spoof_info
    }
    
    if is_spoof:
        print(f"✗ SPOOFING DETECTED for {name}")
        print(f"  Spoof Score: {spoof_score:.3f} (Threshold: {spoof_config['threshold']})")
        print(f"  Texture Score: {spoof_info.get('texture_score', 0):.3f}")
        print(f"  Motion Score: {spoof_info.get('motion_score', 0):.3f}")
        print(f"  Blinks Detected: {spoof_info.get('blinks', 0)}")
        
        # Send SPOOFING status to Proteus
        send_to_proteus(name, "SPOOFING")
        
        # Show spoof detected window
        def show_spoof_window():
            spoof_window = SpoofDetectedWindow(name, spoof_info)
            spoof_window.window.mainloop()
        
        spoof_thread = threading.Thread(target=show_spoof_window, daemon=True)
        spoof_thread.start()
    else:
        print(f"✓ Anti-spoofing PASSED for {name}")
        print(f"  Spoof Score: {spoof_score:.3f} (Threshold: {spoof_config['threshold']})")
    
    return not is_spoof, spoof_info

def mark_all_absent():
    """Mark all unmarked students as absent in CSV"""
    current_time = datetime.now().strftime("%H:%M:%S")
    absent_count = 0
    
    print("\n" + "="*60)
    print("MARKING ALL UNATTENDED STUDENTS AS ABSENT")
    print("="*60)
    
    for name in known_face_names:
        if name not in marked_students:
            # Check if spoof was attempted
            spoof_status = "Spoof_Attempt" if name in spoof_check_results and spoof_check_results[name]['is_spoof'] else "No"
            lnwriter.writerow([name, "Absent", current_time, "No", spoof_status])
            absent_count += 1
            print(f"✗ Marked absent: {name}")
    
    fichier.flush()
    print(f"\n✓ Marked {absent_count} students as absent")
    print(f"Total present: {len(marked_students)}/{len(known_face_names)}")
    
    return absent_count

# ============================================================================
# MAIN PROGRAM
# ============================================================================
print(f"\nSystem started - Timeout: {timeout_minutes} min")
print(f"Anti-Spoofing: {'ENABLED' if spoof_config['enabled'] else 'DISABLED'}")
print(f"Spoof Threshold: {spoof_config['threshold']}")
print(f"Face Detection Timeout: {FACE_DETECTION_TIMEOUT}s")
print(f"RFID Timeout: {RFID_TIMEOUT}s")
print("Waiting for faces...")
print("\n=== CONTROL KEYS ===")
print("  'q' - Quit system (normal exit)")
print("  's' - Stop and mark all absent (early termination)")
print("  'd' - Toggle anti-spoofing detection")
print("="*50)

# Main loop
while system_running:
    ret, frame = video_capture.read()
    if not ret:
        break
        
    current_time = datetime.now()
    
    # Check timeout
    if current_time >= timeout_time:
        print("\n=== TIME OUT ===")
        mark_all_absent()
        break
    
    # Clean up expired pending verifications
    expired_names = []
    for name, detection_info in pending_verification.items():
        # Check what type of pending verification this is
        if isinstance(detection_info, dict):
            if detection_info.get("type") == "rfid" and "rfid_start" in detection_info:
                # RFID verification timeout - use RFID_TIMEOUT
                time_since_rfid = (current_time - detection_info["rfid_start"]).total_seconds()
                if time_since_rfid > RFID_TIMEOUT:
                    expired_names.append(name)
                    print(f"\n✗ RFID timeout for {name} (after {time_since_rfid:.1f}s)")
                    
                    # Send DENIED status for timeout
                    send_to_proteus(name, "DENIED")
                    
                    # Show RFID timeout window
                    def show_timeout_window(name=name):
                        timeout_window = RFIDDeniedWindow(name)
                        timeout_window.window.mainloop()
                    
                    timeout_thread = threading.Thread(target=show_timeout_window, daemon=True)
                    timeout_thread.start()
            else:
                # Other dictionary format (spoof check) - use FACE_DETECTION_TIMEOUT
                detection_time = detection_info.get("face_detected", current_time)
                if isinstance(detection_time, datetime):
                    if (current_time - detection_time).total_seconds() > FACE_DETECTION_TIMEOUT:
                        expired_names.append(name)
                        print(f"\n✗ Face detection timeout for {name}")
                        
                        # Send DENIED status for timeout
                        send_to_proteus(name, "DENIED")
                        
                        # Show RFID timeout window
                        def show_timeout_window(name=name):
                            timeout_window = RFIDDeniedWindow(name)
                            timeout_window.window.mainloop()
                        
                        timeout_thread = threading.Thread(target=show_timeout_window, daemon=True)
                        timeout_thread.start()
        else:
            # Old format (datetime object for face detection) - use FACE_DETECTION_TIMEOUT
            detection_time = detection_info
            if (current_time - detection_time).total_seconds() > FACE_DETECTION_TIMEOUT:
                expired_names.append(name)
                print(f"\n✗ Face detection timeout for {name}")
                
                # Send DENIED status for timeout
                send_to_proteus(name, "DENIED")
                
                # Show RFID timeout window
                def show_timeout_window(name=name):
                    timeout_window = RFIDDeniedWindow(name)
                    timeout_window.window.mainloop()
                
                timeout_thread = threading.Thread(target=show_timeout_window, daemon=True)
                timeout_thread.start()
    
    for name in expired_names:
        if name in pending_verification:
            del pending_verification[name]
        if name == current_verification_target:
            current_verification_target = None
            rfid_window_active = False
    
    # Check for RFID input from GUI
    if not rfid_input_queue.empty():
        rfid_data = rfid_input_queue.get()
        
        if len(rfid_data) == 3:  # With spoof flag
            rfid_value, target_name, spoof_passed = rfid_data
        else:
            rfid_value, target_name = rfid_data
            spoof_passed = True  # Default if not provided
        
        if rfid_value == "TIMEOUT":
            print(f"\n✗ RFID window timeout for {target_name}")
            if target_name in pending_verification:
                del pending_verification[target_name]
            
            # Send DENIED status
            send_to_proteus(target_name, "DENIED")
            
            # Show timeout window
            def show_timeout_window(name=target_name):
                timeout_window = RFIDDeniedWindow(name)
                timeout_window.window.mainloop()
            
            timeout_thread = threading.Thread(target=show_timeout_window, daemon=True)
            timeout_thread.start()
            
            rfid_window_active = False
            current_verification_target = None
            
        elif rfid_value == "CANCELLED":
            print(f"\n✗ RFID verification cancelled for {target_name}")
            if target_name in pending_verification:
                del pending_verification[target_name]
            
            rfid_window_active = False
            current_verification_target = None
            
        elif rfid_value and target_name:
            # Verify RFID
            if verify_rfid(target_name, rfid_value):
                if target_name in students:
                    students.remove(target_name)
                marked_students.add(target_name)
                current_time_str = current_time.strftime("%H:%M:%S")
                print(f"\n✓ Verified: {target_name}")
                
                if target_name in pending_verification:
                    del pending_verification[target_name]
                
                # Show access granted window
                def show_access_granted_window(name=target_name):
                    granted_window = AccessGrantedWindow(name)
                    granted_window.window.mainloop()
                
                granted_thread = threading.Thread(target=show_access_granted_window, daemon=True)
                granted_thread.start()
                
                # Send ACCESS status to Proteus
                send_to_proteus(target_name, f"Hi {target_name}")
                
                # Write to CSV with spoof status
                spoof_status = "Passed" if spoof_passed else "Failed"
                lnwriter.writerow([target_name, "Present", current_time_str, "Yes", spoof_status])
                fichier.flush()
            else:
                print(f"\n✗ Wrong RFID for {target_name}")
                
                if target_name in pending_verification:
                    del pending_verification[target_name]
                
                # Show RFID denied window
                def show_rfid_denied_window(name=target_name):
                    denied_window = RFIDDeniedWindow(name)
                    denied_window.window.mainloop()
                
                denied_thread = threading.Thread(target=show_rfid_denied_window, daemon=True)
                denied_thread.start()
                
                # Send RFID_WRONG status
                send_to_proteus(target_name, "RFID_WRONG")
            
            rfid_window_active = False
            current_verification_target = None
    
    # Face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    if process_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            
            face_names.append(name)
            
            can_detect = True
            if name != "Unknown" and name in last_face_detection_time:
                time_since_last = (current_time - last_face_detection_time[name]).total_seconds()
                if time_since_last < face_detection_cooldown:
                    can_detect = False
            
            # Check cooldown for unknown faces
            unknown_cooldown_ok = True
            if name == "Unknown" and last_unknown_detection:
                time_since_unknown = (current_time - last_unknown_detection).total_seconds()
                if time_since_unknown < unknown_face_cooldown:
                    unknown_cooldown_ok = False
            
            # Handle unknown face
            if name == "Unknown" and unknown_cooldown_ok:
                last_unknown_detection = current_time
                print("\n✗ Unknown face detected")
                
                # Send DENIED status for unknown face to Proteus
                send_to_proteus("Unknown", "DENIED")
            
            elif (name in students and 
                  name not in marked_students and 
                  name not in pending_verification and
                  can_detect and
                  not rfid_window_active and
                  not spoof_check_active):
                
                last_face_detection_time[name] = current_time
                
                # Scale face location to original frame size
                scaled_face_location = tuple(coord * 4 for coord in face_locations[face_names.index(name)])
                
                if spoof_config["enabled"]:
                    # Start anti-spoofing check - DON'T SEND ANYTHING TO PROTEUS YET
                    spoof_check_active = True
                    current_spoof_check_target = name
                    spoof_check_start_time = current_time
                    
                    print(f"\n🔍 Starting anti-spoofing check for {name}...")
                    print("⚠️ NOT sending SCANNING to Proteus yet - waiting for spoof check result")
                    
                    # Add to pending verification (will be removed after spoof check)
                    pending_verification[name] = {
                        "type": "spoof_check",
                        "face_detected": current_time
                    }
                    
                    # Create spoof check window
                    def create_spoof_window():
                        try:
                            spoof_gui = SpoofCheckWindow(name)
                            spoof_gui.window.mainloop()
                        except Exception as e:
                            print(f"Spoof window error: {e}")
                    
                    # Start spoof window in a separate thread
                    spoof_thread = threading.Thread(target=create_spoof_window, daemon=True)
                    spoof_thread.start()
                    
                else:
                    # Skip anti-spoofing, go directly to RFID
                    # Store as dict with RFID start time
                    pending_verification[name] = {
                        "type": "rfid",
                        "rfid_start": current_time,
                        "face_detected": current_time
                    }
                    
                    print(f"\nFace detected: {name}")
                    print("Opening RFID window...")
                    
                    # Send SCANNING status to Proteus
                    # Only send when anti-spoofing is disabled
                    send_to_proteus(name, "SCANNING")
                    
                    # Create RFID window in a separate thread
                    rfid_window_active = True
                    current_verification_target = name
                    
                    # Create and run RFID window
                    def create_rfid_window():
                        try:
                            rfid_gui = RFIDWindow(name, spoof_verified=False)  # No spoof check
                            rfid_gui.window.mainloop()
                        except Exception as e:
                            print(f"RFID window error: {e}")
                            rfid_window_active = False
                    
                    # Start RFID window in a separate thread
                    window_thread = threading.Thread(target=create_rfid_window, daemon=True)
                    window_thread.start()
        
        # Check ongoing spoof detection
        if spoof_check_active and current_spoof_check_target and spoof_check_start_time:
            elapsed_time = (current_time - spoof_check_start_time).total_seconds()
            
            if elapsed_time >= spoof_config["check_duration"]:
                # Perform final spoof check
                if current_spoof_check_target in pending_verification:
                    # Find the face location for this student
                    student_index = -1
                    for i, n in enumerate(face_names):
                        if n == current_spoof_check_target:
                            student_index = i
                            break
                    
                    if student_index >= 0:
                        scaled_face_location = tuple(coord * 4 for coord in face_locations[student_index])
                        
                        # Perform spoof check
                        is_real, spoof_info = perform_spoof_check(
                            current_spoof_check_target, 
                            scaled_face_location, 
                            frame
                        )
                        
                        if is_real:
                            # Spoof check passed, proceed to RFID
                            print(f"\n✓ Anti-spoofing PASSED for {current_spoof_check_target}")
                            print("Opening RFID window...")
                            
                            # NOW send SCANNING to Proteus after spoof check passes
                            send_to_proteus(current_spoof_check_target, "SCANNING")
                            
                            # Update pending verification to RFID mode with start time
                            pending_verification[current_spoof_check_target] = {
                                "type": "rfid",
                                "rfid_start": current_time,
                                "face_detected": spoof_check_start_time
                            }
                            
                            # Create RFID window
                            rfid_window_active = True
                            current_verification_target = current_spoof_check_target
                            
                            def create_rfid_window():
                                try:
                                    rfid_gui = RFIDWindow(current_spoof_check_target, spoof_verified=True)
                                    rfid_gui.window.mainloop()
                                except Exception as e:
                                    print(f"RFID window error: {e}")
                                    rfid_window_active = False
                            
                            window_thread = threading.Thread(target=create_rfid_window, daemon=True)
                            window_thread.start()
                        else:
                            # Spoof detected, mark as spoof attempt
                            print(f"\n✗ SPOOFING DETECTED for {current_spoof_check_target}")
                            
                            # Remove from pending verification
                            if current_spoof_check_target in pending_verification:
                                del pending_verification[current_spoof_check_target]
                            
                            # Don't proceed to RFID
                            rfid_window_active = False
                    
                # Reset spoof check state
                spoof_check_active = False
                current_spoof_check_target = None
                spoof_check_start_time = None
    
    process_frame = not process_frame
    
    # Display information on frame
    time_remaining = max(0, (timeout_time - current_time).total_seconds())
    minutes = int(time_remaining // 60)
    seconds = int(time_remaining % 60)
    
    cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Students: {len(students)}/{len(known_face_names)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Present: {len(marked_students)}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    if spoof_check_active and current_spoof_check_target:
        elapsed = (current_time - spoof_check_start_time).total_seconds()
        time_left = max(0, spoof_config["check_duration"] - elapsed)
        cv2.putText(frame, f"Spoof Check: {current_spoof_check_target}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, f"Time left: {int(time_left)}s", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1)
    
    elif current_verification_target:
        cv2.putText(frame, f"Awaiting RFID: {current_verification_target}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        if current_verification_target in pending_verification:
            detection_info = pending_verification[current_verification_target]
            if isinstance(detection_info, dict) and "rfid_start" in detection_info:
                time_in = (current_time - detection_info["rfid_start"]).total_seconds()
                time_left = max(0, RFID_TIMEOUT - time_in)
                cv2.putText(frame, f"RFID Time left: {int(time_left)}s", (10, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1)
    
    # Draw face rectangles with status
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            status = "UNKNOWN"
        elif name == current_spoof_check_target and spoof_check_active:
            color = (255, 165, 0)  # Orange for spoof check in progress
            status = "SPOOF_CHECK"
        elif name in spoof_check_results and spoof_check_results[name]['is_spoof']:
            color = (0, 0, 255)  # Red for spoof detected
            status = "SPOOF"
        elif name in pending_verification:
            # Check what type of pending verification
            pending_info = pending_verification[name]
            if isinstance(pending_info, dict) and pending_info.get("type") == "rfid":
                color = (255, 0, 255)  # Purple for RFID pending
                status = "RFID_PENDING"
            else:
                color = (255, 255, 0)  # Yellow for other pending
                status = "PENDING"
        elif name in marked_students:
            color = (0, 255, 0)  # Green for verified
            status = "VERIFIED"
        elif name in last_face_detection_time:
            time_since = (current_time - last_face_detection_time[name]).total_seconds()
            if time_since < face_detection_cooldown:
                color = (255, 0, 0)  # Blue for cooldown
                status = f"CD:{int(face_detection_cooldown - time_since)}s"
            else:
                color = (255, 165, 0)  # Orange for ready
                status = "READY"
        else:
            color = (255, 165, 0)  # Orange for ready
            status = "READY"
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, status, (left + 6, bottom - 6), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Display frame
    try:
        if frame.shape[0] != cam_height or frame.shape[1] != cam_width:
            frame_resized = cv2.resize(frame, (cam_width, cam_height))
        else:
            frame_resized = frame
        
        display_frame = imgbackground.copy()
        display_frame[cam_y:cam_y + cam_height, cam_x:cam_x + cam_width] = frame_resized
    except Exception as e:
        display_frame = frame
    
    cv2.imshow("Face Attendance System - Anti-Spoofing", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Handle keyboard controls
    if key == ord('q'):
        # Normal quit
        print("\n=== USER QUIT (q) ===")
        print("Exiting without marking absent...")
        break
        
    elif key == ord('s'):
        # Stop and mark all absent
        print("\n=== STOP REQUESTED (s) ===")
        print("Marking all unmarked students as absent...")
        
        # Mark all absent
        absent_count = mark_all_absent()
        
        # Show confirmation message
        cv2.putText(frame, "SYSTEM STOPPED", (200, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Marked {absent_count} as absent", (180, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Update display
        cv2.imshow("Face Attendance System - Anti-Spoofing", display_frame)
        cv2.waitKey(2000)
        
        # Send message to Proteus
        if ser:
            try:
                ser.write(b"SYS:STOPPED\n")
                ser.flush()
            except:
                pass
        
        break
    
    elif key == ord('d'):
        # Toggle anti-spoofing detection
        spoof_config["enabled"] = not spoof_config["enabled"]
        status = "ENABLED" if spoof_config["enabled"] else "DISABLED"
        print(f"\nAnti-spoofing detection {status}")
        
        if spoof_check_active and not spoof_config["enabled"]:
            # Cancel ongoing spoof check
            spoof_check_active = False
            if current_spoof_check_target in pending_verification:
                del pending_verification[current_spoof_check_target]
            current_spoof_check_target = None
            
            # Close spoof window if open
            print("Cancelled ongoing spoof check")

# ============================================================================
# CLEANUP
# ============================================================================
video_capture.release()
cv2.destroyAllWindows()

# Close windows if still open
if rfid_window_active:
    print("Closing RFID window...")
if spoof_check_active:
    print("Closing spoof check window...")

# Final attendance summary
print("\n" + "="*60)
print("ATTENDANCE SUMMARY")
print("="*60)
print(f"Total students: {len(known_face_names)}")
print(f"Present: {len(marked_students)}")
print(f"Absent: {len(known_face_names) - len(marked_students)}")

# Spoof detection summary
spoof_attempts = 0
if spoof_check_results:
    print("\n" + "-"*60)
    print("SPOOF DETECTION SUMMARY")
    print("-"*60)
    for name, result in spoof_check_results.items():
        if result['is_spoof']:
            spoof_attempts += 1
            print(f"✗ {name}: Spoof attempt detected (Score: {result['spoof_score']:.3f})")
    print(f"Total spoof attempts detected: {spoof_attempts}")

print("="*60)

# Close CSV file
fichier.close()

# Close serial connection
if ser:
    ser.close()

print(f"\n✓ System stopped")
print(f"✓ Attendance saved to: attendance_{current_date}.csv")
print(f"✓ Anti-spoofing detection was {'ENABLED' if spoof_config['enabled'] else 'DISABLED'}")
print(f"✓ Final spoof threshold used: {spoof_config['threshold']}")