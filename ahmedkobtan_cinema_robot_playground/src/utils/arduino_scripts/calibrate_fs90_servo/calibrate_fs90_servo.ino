#include <Servo.h>

Servo myServo;

void setup() {
  // We attach the servo on pin 9.
  // 900 is the min pulse (0 degrees), 2100 is the max (120 degrees).
  myServo.attach(9, 900, 2100);

  // Move to the center position (90 degrees in software = 60 degrees physical center)
  // myServo.write(90);

  myServo.write(0);

  Serial.begin(9600);
  Serial.println("Servo Centered. You can now attach the phone mount arm!");
}

void loop() {
  // Stay centered for calibration
  // Sweep from 0 to 120 degrees
  for (int pos = 0; pos <= 180; pos++) {
    myServo.write(pos);
    delay(15);
  }
  for (int pos = 180; pos >= 0; pos--) {
    myServo.write(pos);
    delay(15);
  }
}
