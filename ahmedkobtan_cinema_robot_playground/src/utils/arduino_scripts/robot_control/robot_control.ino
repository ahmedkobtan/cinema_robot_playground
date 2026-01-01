/*
 * Cinema Robot Control Firmware
 * Arduino Uno - Motor and Servo Control
 *
 * Serial Protocol:
 * - M:F:speed    - Move Forward (speed 0-255)
 * - M:B:speed    - Move Backward (speed 0-255)
 * - M:L:speed    - Turn Left (speed 0-255)
 * - M:R:speed    - Turn Right (speed 0-255)
 * - M:S          - Stop
 * - S:pulse      - Set Servo pulse width in microseconds (900-2100)
 *
 * Responses:
 * - OK           - Command executed successfully
 * - ERROR:xxx    - Error message
 */

#include <Servo.h>

// Motor Pins (L298N Driver)
const int enA = 3;  // Enable pin for motor A (PWM)
const int in1 = 2;  // Motor A direction pin 1
const int in2 = 4;  // Motor A direction pin 2
const int enB = 5;  // Enable pin for motor B (PWM)
const int in3 = 7;  // Motor B direction pin 1
const int in4 = 8;  // Motor B direction pin 2

// Servo Pin
const int servoPin = 9;  // PWM pin for servo
const int minPulse = 900;   // FS90 minimum pulse width
const int maxPulse = 2100; // FS90 maximum pulse width
const int centerPulse = 1500; // Center position

Servo phoneServo;

// Serial communication
String inputString = "";
boolean stringComplete = false;

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  inputString.reserve(200);

  // Set motor pins as outputs
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // Initialize motors to stopped state
  stopMotors();

  // Initialize servo
  phoneServo.attach(servoPin, minPulse, maxPulse);
  phoneServo.writeMicroseconds(centerPulse);

  Serial.println("OK"); // Ready signal
}

void loop() {
  // Read serial input
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    inputString += inChar;
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("M:F:")) {
    // Move Forward
    int speed = cmd.substring(4).toInt();
    speed = constrain(speed, 0, 255);
    moveForward(speed);
    Serial.println("OK");

  } else if (cmd.startsWith("M:B:")) {
    // Move Backward
    int speed = cmd.substring(4).toInt();
    speed = constrain(speed, 0, 255);
    moveBackward(speed);
    Serial.println("OK");

  } else if (cmd.startsWith("M:L:")) {
    // Turn Left
    int speed = cmd.substring(4).toInt();
    speed = constrain(speed, 0, 255);
    turnLeft(speed);
    Serial.println("OK");

  } else if (cmd.startsWith("M:R:")) {
    // Turn Right
    int speed = cmd.substring(4).toInt();
    speed = constrain(speed, 0, 255);
    turnRight(speed);
    Serial.println("OK");

  } else if (cmd == "M:S") {
    // Stop
    stopMotors();
    Serial.println("OK");

  } else if (cmd.startsWith("S:")) {
    // Set Servo
    int pulse = cmd.substring(2).toInt();
    pulse = constrain(pulse, minPulse, maxPulse);
    phoneServo.writeMicroseconds(pulse);
    Serial.println("OK");

  } else {
    Serial.print("ERROR:Unknown command: ");
    Serial.println(cmd);
  }
}

void moveForward(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void moveBackward(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void turnLeft(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void turnRight(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void stopMotors() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  analogWrite(enA, 0);
  analogWrite(enB, 0);
}
