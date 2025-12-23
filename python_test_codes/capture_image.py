import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Save the captured image
    cv2.imwrite('captured_image.jpg', frame)
    print("Image captured successfully!")
else:
    print("Error: Failed to capture image")

# Release the camera
cap.release()