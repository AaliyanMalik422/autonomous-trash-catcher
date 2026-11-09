// Host-side closed-loop test of mecanum_locomotion.ino (compile with g++, see run_firmware_test.sh).
#include "Arduino.h"
uint32_t g_micros = 0;
int g_pin[64];
uint32_t g_duty[64];
SerialClass Serial;

#include "../../mecanum_locomotion/mecanum_locomotion.ino"

static float wv[4] = {0, 0, 0, 0};      // true wheel speeds (m/s)
static float encFrac[4] = {0, 0, 0, 0};
static float trueX = 0, trueY = 0;
static const float PLANT_GAIN = 0.9f;   // plant is a bit weaker than the model
static const float TAU = 0.05f;

static void plantStep(float dt) {
  for (int i = 0; i < 4; i++) {
    float mag = 0;
    if (g_duty[MOTOR[i].pwm] > 0)
      mag = ((float)g_duty[MOTOR[i].pwm] / PWM_MAX - MIN_DUTY) / (1.0f - MIN_DUTY);
    float u = g_pin[MOTOR[i].in1] == HIGH ? mag : -mag;
    if (g_pin[MOTOR[i].in1] == LOW && g_pin[MOTOR[i].in2] == LOW) u = 0;
    if (MOTOR_INVERT[i]) u = -u;
    wv[i] += (u * MAX_WHEEL_MPS * PLANT_GAIN - wv[i]) * dt / TAU;
    encFrac[i] += wv[i] * dt / (2.0f * PI * WHEEL_RADIUS_M) * ENC_COUNTS_PER_WHEEL_REV;
    int whole = (int)encFrac[i];
    encFrac[i] -= whole;
    encCount[i] += whole;
  }
  trueY += 0.25f * (wv[0] + wv[1] + wv[2] + wv[3]) * dt;
  trueX += 0.25f * (wv[0] - wv[1] - wv[2] + wv[3]) * dt;
}

static float run(float seconds, float *maxSpeed = nullptr) {
  int steps = (int)(seconds * 1000);
  for (int k = 0; k < steps; k++) {
    g_micros += 1000;
    plantStep(0.001f);
    loop();
    if (maxSpeed) *maxSpeed = fmaxf(*maxSpeed, hypotf(odoVX, odoVY));
  }
  return 0;
}

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok:   %s\n", msg); } while (0)

int main() {
  setup();
  run(0.2f);

  Serial.feed("<POS,0.300,0.400>\n");
  float vmax = 0;
  float maxOvershoot = 0;
  for (int k = 0; k < 3000; k++) {            // 3 s
    run(0.001f, &vmax);
    maxOvershoot = fmaxf(maxOvershoot, hypotf(trueX, trueY) - 0.5f);
  }
  printf("after POS 0.3,0.4: true=(%.3f,%.3f) odo=(%.3f,%.3f) vmax=%.2f overshoot=%.3f\n",
         trueX, trueY, posX, posY, vmax, maxOvershoot);
  CHECK(hypotf(trueX - 0.3f, trueY - 0.4f) < 0.04f, "reaches position target within 4 cm");
  CHECK(hypotf(posX - trueX, posY - trueY) < 0.03f, "odometry matches true position");
  CHECK(maxOvershoot < 0.03f, "no significant overshoot");
  CHECK(Serial.tx.find("<ACK,POS,0.300,0.400>") != std::string::npos, "ACK sent");
  CHECK(Serial.tx.find("<ODO,") != std::string::npos, "odometry telemetry streamed");

  Serial.feed("<POS,-0.2,0.1>\n");
  run(2.5f);
  CHECK(hypotf(trueX + 0.2f, trueY - 0.1f) < 0.04f, "second target (strafe left + back)");

  Serial.feed("<HOME>\n");
  run(3.0f);
  CHECK(hypotf(trueX, trueY) < 0.04f, "HOME returns to origin");

  Serial.feed("<POS,0.4,0\n<POS,abc,1>\ngarbage\n<POS,nan,1>\n");
  run(0.3f);
  CHECK(Serial.tx.find("<ERR,POS>") != std::string::npos, "malformed POS rejected");
  CHECK(hypotf(trueX, trueY) < 0.04f, "bad commands do not move the robot");

  Serial.feed("<POS,9,9>\n");
  run(0.01f);
  CHECK(goalX <= POS_LIMIT_M + 1e-6f && goalY <= POS_LIMIT_M + 1e-6f, "far targets clamped to limit");
  Serial.feed("<STOP>\n");
  run(1.5f);
  float sx = trueX, sy = trueY;
  run(0.5f);
  CHECK(hypotf(trueX - sx, trueY - sy) < 0.01f, "STOP holds position");

  Serial.feed("<ZERO>\n");
  run(0.05f);
  CHECK(fabsf(posX) < 1e-4f && fabsf(posY) < 1e-4f, "ZERO resets odometry");

  // control loop must keep running while a line is only half received (no blocking)
  Serial.feed("<POS,0.1");
  uint32_t t0 = g_micros;
  run(0.5f);
  CHECK(g_micros - t0 == 500000, "half-received line does not block loop");

  printf(fails ? "\n%d FAILED\n" : "\nALL PASSED\n", fails);
  return fails ? 1 : 0;
}
