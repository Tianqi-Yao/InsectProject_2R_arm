#include <Servo.h>
#include "kinematics.h"

// --- Hardware ---
#define PIN_SERVO1  9
#define PIN_SERVO2  10
#define BAUD_RATE   115200

// --- Motion interpolation ---
// Max degrees per step; smaller = smoother but slower
#define INTERP_STEP   1.0f
// Delay between steps (ms); tune for speed vs. smoothness
#define INTERP_DELAY  10

Servo servo1;
Servo servo2;

// Home position: workspace centre (100mm, 75mm) → arm-relative (0, 120mm)
// IK result: theta1≈44.4°, theta2≈115.6° → s1≈67.5°, s2≈115.6°
#define HOME_S1  68.0f
#define HOME_S2 116.0f

// Current servo positions (degrees)
float cur_s1 = HOME_S1;
float cur_s2 = HOME_S2;

// Serial input buffer
char buf[64];
uint8_t buf_idx = 0;

// ---------------------------------------------------------------
// Move servos smoothly from current position to target
// ---------------------------------------------------------------
void smooth_move(float target_s1, float target_s2) {
    float d1 = target_s1 - cur_s1;
    float d2 = target_s2 - cur_s2;
    float dist = max(fabs(d1), fabs(d2));
    int steps = (int)(dist / INTERP_STEP) + 1;

    for (int i = 1; i <= steps; i++) {
        float t = (float)i / steps;
        int s1 = (int)constrain(cur_s1 + d1 * t, 0.0f, 180.0f);
        int s2 = (int)constrain(cur_s2 + d2 * t, 0.0f, 180.0f);
        servo1.write(s1);
        servo2.write(s2);
        delay(INTERP_DELAY);
    }
    cur_s1 = target_s1;
    cur_s2 = target_s2;
}

// ---------------------------------------------------------------
// Process a complete line from serial
// Protocol:
//   "G X<float> Y<float>\n"  → move to (x, y) in mm
//   "H\n"                    → home (arm points straight, mid-range)
//   "W\n"                    → where (print current servo angles)
// ---------------------------------------------------------------
void process_command(const char *line) {
    if (line[0] == 'G' || line[0] == 'g') {
        float x = 0.0f, y = 0.0f;
        if (sscanf(line + 1, " X%f Y%f", &x, &y) != 2) {
            Serial.println("ERR: BAD_FORMAT  (expected: G X<mm> Y<mm>)");
            return;
        }
        IKResult r = ik_solve(x, y);
        if (!r.reachable) {
            Serial.print("ERR: OUT_OF_RANGE (");
            Serial.print(x); Serial.print(", "); Serial.print(y);
            Serial.println(")");
            return;
        }
        smooth_move(r.servo1, r.servo2);
        Serial.print("OK theta1="); Serial.print(r.theta1, 1);
        Serial.print(" theta2=");   Serial.print(r.theta2, 1);
        Serial.print(" s1=");       Serial.print(r.servo1, 1);
        Serial.print(" s2=");       Serial.println(r.servo2, 1);

    } else if (line[0] == 'H' || line[0] == 'h') {
        // Home: workspace centre (75, 100) mm
        smooth_move(HOME_S1, HOME_S2);
        Serial.println("OK HOME");

    } else if (line[0] == 'W' || line[0] == 'w') {
        Serial.print("POS s1="); Serial.print(cur_s1, 1);
        Serial.print(" s2=");    Serial.println(cur_s2, 1);

    } else {
        Serial.print("ERR: UNKNOWN_CMD ("); Serial.print(line); Serial.println(")");
    }
}

// ---------------------------------------------------------------
void setup() {
    Serial.begin(BAUD_RATE);
    servo1.attach(PIN_SERVO1);
    servo2.attach(PIN_SERVO2);

    // Move to home position on startup (workspace centre)
    servo1.write((int)HOME_S1);
    servo2.write((int)HOME_S2);
    delay(500);

    Serial.println("2R_ARM READY");
    Serial.println("Commands: G X<mm> Y<mm> | H (home) | W (where)");
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (buf_idx > 0) {
                buf[buf_idx] = '\0';
                process_command(buf);
                buf_idx = 0;
            }
        } else if (buf_idx < sizeof(buf) - 1) {
            buf[buf_idx++] = c;
        }
    }
}
