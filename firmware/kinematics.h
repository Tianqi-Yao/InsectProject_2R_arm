#pragma once
#include <math.h>

// 2R planar arm inverse kinematics (horizontal plane, elbow-up configuration)
// All units: mm for lengths, degrees for angles
// Coordinate origin: shoulder joint (servo1 axis)
// +Y points away from base, +X points right

#define IK_L1 125.0f
#define IK_L2  95.0f

// Workspace limits relative to base (mm)
#define IK_MAX_REACH (IK_L1 + IK_L2)           // 250mm
#define IK_MIN_REACH (fabs(IK_L1 - IK_L2))     // 0mm (equal links)

// Servo angle mapping: physical servo range 0~180 deg
//
// Workspace: 200mm (X) × 150mm (Y), base at (100, -45) in workspace coords.
// Derivation (L1=125mm, L2=95mm, base 45mm behind 200mm near edge):
//   near-right corner arm-relative: (100, 45), d = sqrt(100²+45²) = 109.66mm
//   c2 = (109.66²-125²-95²)/(2×125×95) = -0.5316, s2 = 0.8470
//   beta = atan2(95×0.8470, 125+95×(-0.5316)) = atan2(80.47, 74.50) = 47.18°
//   theta1_near_right = atan2(45,100) - 47.18° = -22.98°
//   → SERVO1_OFFSET = 23.08 maps theta1_min to servo ~0.1°
//   theta2 ranges from +10° (far corners) to +162° (near centre)
//   → SERVO2_OFFSET = 0; apply 5° software margin in firmware or PC side
//
// Fine-tune both values during physical calibration.
#define SERVO1_OFFSET  23.08f  // servo1_cmd = theta1 + SERVO1_OFFSET
#define SERVO2_OFFSET   0.0f   // servo2_cmd = theta2 + SERVO2_OFFSET

struct IKResult {
    float theta1;   // shoulder angle (degrees, math convention: 0=+X, CCW positive)
    float theta2;   // elbow angle   (degrees, positive = elbow-up/left)
    float servo1;   // servo1 command (degrees, 0~180)
    float servo2;   // servo2 command (degrees, 0~180)
    bool  reachable;
};

// Returns true and fills result when (x,y) is within workspace.
// theta1/theta2 are in standard math angles (degrees).
// servo1/servo2 are mapped to 0~180 deg for the Servo library.
inline IKResult ik_solve(float x, float y, float L1 = IK_L1, float L2 = IK_L2) {
    IKResult r;
    r.reachable = false;

    float d2 = x * x + y * y;
    float c2 = (d2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);

    // Check reachability
    if (c2 < -1.0f || c2 > 1.0f) return r;

    // Elbow-up: theta2 > 0 (arm bends consistently to one side)
    float s2 = sqrtf(1.0f - c2 * c2);
    r.theta2 = atan2f(s2, c2) * 180.0f / M_PI;

    // theta1: direction to target minus half of theta2 (L1=L2 simplification holds exactly)
    float alpha = atan2f(y, x) * 180.0f / M_PI;
    float beta  = atan2f(L2 * s2, L1 + L2 * c2) * 180.0f / M_PI;
    r.theta1 = alpha - beta;

    // Map to servo range [0, 180]
    r.servo1 = r.theta1 + SERVO1_OFFSET;
    r.servo2 = r.theta2 + SERVO2_OFFSET;

    // Validate servo range
    if (r.servo1 < 0.0f || r.servo1 > 180.0f) return r;
    if (r.servo2 < 0.0f || r.servo2 > 180.0f) return r;

    r.reachable = true;
    return r;
}

// Forward kinematics: given joint angles (degrees, math convention), compute end-effector (x,y)
inline void fk_solve(float theta1_deg, float theta2_deg,
                     float &ex, float &ey,
                     float L1 = IK_L1, float L2 = IK_L2) {
    float t1 = theta1_deg * M_PI / 180.0f;
    float t2 = theta2_deg * M_PI / 180.0f;
    float ex1 = L1 * cosf(t1);
    float ey1 = L1 * sinf(t1);
    ex = ex1 + L2 * cosf(t1 + t2);
    ey = ey1 + L2 * sinf(t1 + t2);
}
