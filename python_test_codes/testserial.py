import serial
try:
    # Try to create a serial object
    ser = serial.Serial('COM1', 9600, timeout=1)
    print("Serial port opened successfully!")
    ser.close()
except serial.SerialException as e:
    print(f"Serial port error: {e}")
except Exception as e:
    print(f"Other error: {e}")
