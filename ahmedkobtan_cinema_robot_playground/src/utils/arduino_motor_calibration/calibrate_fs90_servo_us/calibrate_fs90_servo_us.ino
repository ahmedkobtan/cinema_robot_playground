#include <Servo.h>

Servo phoneServo;

const int servoPin = 9; // Using PWM Pin 9
const int minPulse = 900;  // From FS90 specs
const int maxPulse = 2100; // From FS90 specs
const int centerPulse = 1500; // The 60-degree mid-point

void setup() {
  Serial.begin(9600);
  phoneServo.attach(servoPin, minPulse, maxPulse);

  Serial.println("Moving to CENTER (1500us)...");
  phoneServo.writeMicroseconds(centerPulse);
  delay(2000);
}

void loop() {
  // Simple sweep test to verify the 120-degree range
  Serial.println("Sweeping to 900us...");
  phoneServo.writeMicroseconds(minPulse);
  delay(2000);

  Serial.println("Moving back to CENTER (1500us)...");
  phoneServo.writeMicroseconds(centerPulse);
  delay(2000);

  Serial.println("Sweeping to 2100us...");
  phoneServo.writeMicroseconds(maxPulse);
  delay(2000);
}
