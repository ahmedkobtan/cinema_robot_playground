#include <Servo.h>

// --- PIN DEFINITIONS ---
const int servoPin = 9;
// Motor A (Left)
const int enA = 3;
const int in1 = 2;
const int in2 = 4;
// Motor B (Right)
const int enB = 5;
const int in3 = 7;
const int in4 = 8;

// --- SERVO SETTINGS ---
Servo phoneServo;
const int servoMin = 900;
const int servoMax = 2100;
const int servoCenter = 1500;

void setup() {
  // Initialize Pins
  pinMode(enA, OUTPUT); pinMode(in1, OUTPUT); pinMode(in2, OUTPUT);
  pinMode(enB, OUTPUT); pinMode(in3, OUTPUT); pinMode(in4, OUTPUT);

  phoneServo.attach(servoPin, servoMin, servoMax);
  phoneServo.writeMicroseconds(servoCenter); // Start at side-facing position

  delay(2000); // Wait for you to put it down
}

void loop() {
  // --- CINEMATIC TRACKING SHOT ---
  // Drive forward slowly while panning the camera
  moveForward(200);

  // Smoothly pan from 1200us to 1800us
  for (int pos = 1200; pos <= 1800; pos += 5) {
    phoneServo.writeMicroseconds(pos);
    delay(20);
  }

  stopMotors();
  delay(1000);

  // Drive backward while panning back
  moveBackward(200);
  for (int pos = 1800; pos >= 1200; pos -= 5) {
    phoneServo.writeMicroseconds(pos);
    delay(20);
  }

  stopMotors();
  phoneServo.writeMicroseconds(servoCenter);

  while(1); // End of test
}

// --- HELPER FUNCTIONS ---
void moveForward(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, HIGH); digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH); digitalWrite(in4, LOW);
}

void moveBackward(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, LOW); digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW); digitalWrite(in4, HIGH);
}

void stopMotors() {
  digitalWrite(in1, LOW); digitalWrite(in2, LOW);
  digitalWrite(in3, LOW); digitalWrite(in4, LOW);
}
