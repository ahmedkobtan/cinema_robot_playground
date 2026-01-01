import time

import serial

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/ttyACM0"  # This is usually the Prusa. If it fails, try /dev/ttyUSB0
BAUD_RATE = 115200


def connect_robot(serial_port=SERIAL_PORT):
    try:
        ser = serial.Serial(serial_port, BAUD_RATE, timeout=1)
        print(f"Connected to {serial_port} @ {BAUD_RATE}")
        time.sleep(2)  # Wait for connection to settle
        # Clear startup garbage
        while ser.in_waiting:
            print(f"Robot says: {ser.readline().decode().strip()}")
        return ser
    except Exception as e:
        print(f"Error connecting: {e}")
        return None


def send_command(ser, command):
    # Remove comments (everything after ;) and whitespace
    clean_cmd = command.split(";")[0].strip()
    if not clean_cmd:
        return

    print(f"Sending: {clean_cmd}")
    ser.write((clean_cmd + "\n").encode())

    # Wait for OK
    while True:
        line = ser.readline().decode().strip()
        if line:
            print(f"  < {line}")
        if "ok" in line.lower():
            break


def print_file(ser, filename):
    print(f"--- Printing file: {filename} ---")
    try:
        with open(filename, "r") as f:
            for line in f:
                send_command(ser, line)
        print("--- Print Complete! ---")
    except FileNotFoundError:
        print("Error: File not found!")


if __name__ == "__main__":
    # --- MAIN LOOP ---
    ser = connect_robot()
    if ser:
        print("\n--- ROBOT COMMANDER READY ---")
        print("Type a G-Code command (e.g., G28) or 'print filename.gcode'")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("Cmd > ")

            if user_input.lower() == "exit":
                ser.close()
                break

            if user_input.startswith("print "):
                filename = user_input.split(" ")[1]
                print_file(ser, filename)
            else:
                send_command(ser, user_input)
