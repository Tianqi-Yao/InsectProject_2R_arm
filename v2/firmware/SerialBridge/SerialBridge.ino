// Minimal USB <-> servo-bus bridge for the Waveshare ESP32 servo driver board.
//
// No WiFi, no display, no servo-specific logic of its own -- it just relays
// raw bytes between the USB serial port (to the Raspberry Pi) and the
// STS3215 bus (Serial1), so the Pi can speak the SCServo protocol directly
// via scservo_sdk over USB, exactly as if the ESP32 weren't there at all.
// This is the firmware for normal operation; ServoJog.ino (the WiFi jog
// tool) is for manual wiring/ID checks and should not be flashed at the
// same time as this one -- they both want exclusive control of Serial1.
//
// Wiring is unchanged: servo bus on Serial1, RX=GPIO18, TX=GPIO19.
//
// USB_BAUD and SERVO_BAUD are intentionally different: the two UARTs are
// independent hardware peripherals, nothing requires them to match. The
// servo bus must stay at 1,000,000 (the STS3215's default), but 1,000,000
// on the *USB* side turned out to be unreliable over this board's USB-serial
// chip on macOS (a bare echo test at 1M baud was dropping bytes mid-packet).
// 115200 is a standard, safe rate and plenty fast for these small packets.

#define SERVO_RXD 18
#define SERVO_TXD 19
#define SERVO_BAUD 1000000
#define USB_BAUD 115200

void setup() {
  Serial.begin(USB_BAUD);
  Serial1.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RXD, SERVO_TXD);
}

void loop() {
  while (Serial.available()) {
    Serial1.write(Serial.read());
  }
  while (Serial1.available()) {
    Serial.write(Serial1.read());
  }
}
