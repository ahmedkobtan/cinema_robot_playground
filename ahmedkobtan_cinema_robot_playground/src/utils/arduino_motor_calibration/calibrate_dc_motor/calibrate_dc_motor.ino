#include <Servo.h>

// Motor Pins
const int enA = 3;
const int in1 = 2;
const int in2 = 4;
const int enB = 5;
const int in3 = 7;
const int in4 = 8;

void setup() {
  // Set all motor pins as outputs
  pinMode(enA, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
}

void loop() {
  // --- TEST FORWARD (5 Seconds) ---
  moveForward(150); // Speed 0-255
  delay(5000);

  // --- STOP (2 Seconds) ---
  stopMotors();
  delay(2000);

  // --- TEST BACKWARD (5 Seconds) ---
  moveBackward(150);
  delay(5000);

  stopMotors();
  while(1); // Stop loop after one test
}

void moveForward(int speed) {
  analogWrite(enA, speed);
  analogWrite(enB, speed);
  digitalWrite(in1, HIGH); digitalWrite(in2, LOW); // Left
  digitalWrite(in3, HIGH); digitalWrite(in4, LOW); // Right
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
