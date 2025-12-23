# 🚀 RFID-Attendance System with Face Verification

A robust and secure attendance management solution combining RFID technology with real-time face detection for enhanced verification.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-Apache_2.0-green)
![Stars](https://img.shields.io/github/stars/hedi0/RFID-Attendance?style=social)
![Forks](https://img.shields.io/github/forks/hedi0/RFID-Attendance?style=social)

![Project Preview](/images/placeholder_preview.png)
_Placeholder: A screenshot or diagram showcasing the system in action._

---

## ✨ Features

This project offers a comprehensive and secure approach to attendance tracking:

*   ✨ **Dual Authentication:** Secure attendance tracking using both RFID card scans and real-time facial recognition for accurate user verification.
*   ⚡ **Real-time Processing:** Instantly record attendance events with quick RFID reads and efficient face detection algorithms, minimizing delays.
*   📈 **Scalable & Flexible:** Designed to easily integrate with various RFID readers and adapt to different attendance logging requirements and environments.
*   🐍 **Python-Powered:** Built entirely in Python, offering high extensibility, ease of customization, and a broad range of available libraries.
*   📸 **Visual Verification:** Capture and store images during attendance events, providing a visual log for future audit and verification purposes.

---

## 🛠️ Installation Guide

Follow these steps to get your RFID-Attendance system up and running.

### Prerequisites

Ensure you have the following installed on your system:

*   **Python 3.x** (recommended Python 3.8+)
*   **pip** (Python package installer)
*   An **RFID Reader** connected via serial port (e.g., USB-to-UART converter)
*   A **Webcam** for face detection

### Step-by-Step Installation

1.  **Clone the Repository:**
    Start by cloning the project repository to your local machine:

    ```bash
    git clone https://github.com/hedi0/RFID-Attendance.git
    cd RFID-Attendance
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage project dependencies:

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**

    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
    _Note: A `requirements.txt` file is assumed. If not present, you might need to install individual packages like `opencv-python`, `numpy`, `pyserial`, `face_recognition` manually._
    Example for manual installation:
    ```bash
    pip install opencv-python numpy pyserial face_recognition
    ```

5.  **Configure the System:**
    Open `code.py` in your preferred editor and make the following adjustments:
    *   **Serial Port:** Update the serial port name for your RFID reader (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux).
    *   **User Data:** Populate the user database (e.g., a dictionary mapping RFID UIDs to names, and known face encodings).
    *   **Image Paths:** Ensure the `images` directory exists for storing captured photos and that `background_image.jpg` is correctly referenced.

---

## 🚀 Usage Examples

Once installed and configured, you can run the attendance system.

### Basic Execution

To start the RFID-Attendance system, simply run the main Python script:

```bash
python code.py
```

Upon execution, the system will:
1.  Initialize the webcam for face detection.
2.  Start listening for RFID card scans.
3.  When an RFID card is scanned:
    *   It will attempt to identify the user.
    *   Simultaneously, it will use the webcam to detect and verify the user's face.
    *   If both RFID and face verification pass, the attendance will be recorded.
    *   A confirmation message will be displayed, and an image might be captured.

### Example Scenario

Imagine a user named "John Doe" with RFID tag `12345678`. When John scans his card and stands in front of the camera:

*   The system reads `12345678`.
*   It identifies John Doe from the RFID tag.
*   It then detects John's face via the webcam and matches it against pre-registered face encodings for John Doe.
*   Upon successful match, "John Doe's attendance recorded!" is displayed, and a timestamped image might be saved.

![Usage Screenshot](/images/usage_screenshot.png)
_Placeholder: A screenshot showing the system's output during an attendance event._

---

## 🗺️ Project Roadmap

We have exciting plans for the future development of the RFID-Attendance system:

*   **Version 1.1 - Database Integration:**
    *   Implement a lightweight database (e.g., SQLite) for persistent storage of attendance logs and user data.
    *   Allow dynamic adding/removing of users without code modification.
*   **Version 1.2 - Web Interface & Reporting:**
    *   Develop a basic web interface using Flask or Django for real-time monitoring and attendance reporting.
    *   Generate daily/monthly attendance reports.
*   **Version 1.3 - Enhanced Face Recognition:**
    *   Explore more robust and faster face recognition models.
    *   Add liveness detection to prevent spoofing attempts.
*   **Future Enhancements:**
    *   Support for multiple RFID readers.
    *   Integration with external APIs (e.g., for sending notifications).
    *   Containerization (Docker) for easier deployment.

---

## 🤝 Contribution Guidelines

We welcome contributions from the community to make this project even better! Please follow these guidelines:

### Code Style

*   Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style.
*   Use clear and descriptive variable and function names.
*   Include comments for complex logic.

### Branch Naming Conventions

*   **Features:** Use `feature/your-feature-name` (e.g., `feature/database-integration`).
*   **Bug Fixes:** Use `bugfix/issue-description` (e.g., `bugfix/serial-port-error`).
*   **Hotfixes:** Use `hotfix/critical-bug-fix`.

### Pull Request Process

1.  **Fork** the repository.
2.  **Create a new branch** from `main` (or `develop` if present) using the naming conventions above.
3.  **Make your changes**, ensuring they align with the project's goals.
4.  **Commit your changes** with clear and concise commit messages.
5.  **Push your branch** to your forked repository.
6.  **Open a Pull Request** (PR) to the `main` branch of the original repository.
7.  Provide a clear description of your changes and why they are needed.

### Testing Requirements

*   Ensure your changes do not introduce new bugs.
*   If applicable, add basic functional tests for new features.
*   Test your changes thoroughly in your local environment before submitting a PR.

---

## 📄 License Information

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software under the terms of the Apache License 2.0.

For more details, see the [LICENSE](LICENSE) file in the repository or visit the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) official page.

Copyright (c) 2023 hedi0. All rights reserved.
