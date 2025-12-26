# 🚀 RFID-Attendance System with Face Verification

A robust and secure attendance management solution combining **RFID technology** with **real-time face detection** for enhanced verification.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache_2.0-green)
![Stars](https://img.shields.io/github/stars/hedi0/RFID-Attendance?style=social)
![Forks](https://img.shields.io/github/forks/hedi0/RFID-Attendance?style=social)

![Project Cover](/images/rfid_att_cover.png)

---

## ✨ Features

* **Dual Authentication:** RFID card scans + real-time facial recognition for accurate verification.
* **Real-time Processing:** Instant attendance recording with fast RFID reads and efficient face detection.
* **Scalable & Flexible:** Works with various RFID readers and adapts to different environments.
* **Python-Powered:** Fully built in Python for easy customization and extensibility.
* **Visual Verification:** Captures images during attendance events for auditing and verification.

---

## 🛠️ Installation Guide

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* RFID Reader (connected via serial port, e.g., USB-to-UART)
* Webcam for face detection
* **Proteus Simulation:** Ensure Proteus is installed and ready for hardware simulation.

### Step-by-Step Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/hedi0/RFID-Attendance.git
   cd RFID-Attendance
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   * **Windows:** `.env\Scripts\activate`
   * **macOS/Linux:** `source venv/bin/activate`

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is missing, install manually:*

   ```bash
   pip install opencv-python numpy pyserial face_recognition
   ```

5. **Configure the System:**

   * Update the **serial port** for your RFID reader in `code.py` (`COM3` on Windows, `/dev/ttyUSB0` on Linux).
   * Populate **user data** (RFID UIDs → names, known face encodings).
   * Ensure the `images` directory exists for captured photos and `background_image.jpg` is correctly referenced.

---

## 🚀 Usage

### Run the System

```bash
python code.py
```

> **Note:** Proteus simulation should be running simultaneously to emulate the hardware (RFID reader and connected components).

**Workflow:**

1. Webcam initializes for face detection.
2. System listens for RFID card scans.
3. When a card is scanned:

   * Identify the user via RFID.
   * Detect and verify the user's face via webcam.
   * If both match, attendance is recorded and a confirmation message is displayed.
   * Optionally, an image of the event is saved.

### Example Scenario

User **John Doe** has RFID tag `12345678`:

* Card scanned → system identifies John.
* Face verified → "John Doe's attendance recorded!"
* Timestamped image saved for audit.

![Usage Screenshot](/images/usage_screenshot.png)
*Placeholder: System output during an attendance event.*

---

## 🗺️ Roadmap

* **v1.1 – Database Integration:** SQLite for persistent logs and dynamic user management.
* **v1.2 – Web Interface:** Real-time monitoring, reporting via Flask/Django.
* **v1.3 – Enhanced Face Recognition:** Faster models, liveness detection.
* **Future Enhancements:** Multiple RFID readers, API integration, Docker containerization.

---

## 🤝 Contribution

### Guidelines

* Follow **PEP 8** for Python style.
* Use descriptive names and comments.

### Branch Naming

* Features: `feature/your-feature-name`
* Bugfixes: `bugfix/issue-description`
* Hotfixes: `hotfix/critical-bug-fix`

### Pull Requests

1. Fork → create a branch → make changes → commit → push → open PR
2. Provide clear description and reason for changes.

### Testing

* Ensure no new bugs.
* Add basic functional tests for new features.

---

## 📄 License

**Apache License 2.0** – see [LICENSE](LICENSE) for details.
Copyright (c) 2023 hedi0. All rights reserved.
