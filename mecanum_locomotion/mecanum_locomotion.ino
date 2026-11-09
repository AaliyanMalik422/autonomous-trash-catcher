/*
 * Autonomous Trash Catching Robot - Module 2
 * Mecanum Kinematics, Motor PWM Control and Acceleration Limiting (ESP32)
 *
 * Serial protocol (115200 baud, ASCII, one command per line, see trajectory/esp32_serial.py):
 *   <POS,x,y>      go to (x,y) metres in the robot HOME frame (+x strafe right, +y forward)
 *   <HOME>         go to (0,0)
 *   <STOP>         hold the current position
 *   <ZERO>         declare the current position as (0,0) (re-zero odometry)
 *   <VEL,x,y,w>    manual test: body velocities m/s, m/s, rad/s (times out after 250 ms)
 * Replies:  <ACK,...> per command, and <ODO,x,y,vx,vy> at 20 Hz.
 *
 * Works with ESP32 Arduino core 2.x and 3.x (ledc API differences handled below).
 */

#include <Arduino.h>
#include <math.h>

// ============================ CONFIGURATION =================================

#define USE_ENCODERS 1          // 1 = closed-loop wheel speed PI, 0 = open loop
#define DEMO_PATTERN 0          // 1 = run a built-in test pattern from loop()

// ---- PWM (ledc) ----
constexpr uint32_t PWM_FREQ_HZ  = 20000;   
constexpr uint8_t  PWM_RES_BITS = 10;      
constexpr uint32_t PWM_MAX      = (1u << PWM_RES_BITS) - 1;
constexpr float    MIN_DUTY     = 0.08f;   

// ---- Geometry ----
constexpr float WHEEL_RADIUS_M = 0.0485f;  
constexpr float HALF_WHEELBASE = 0.10f;    
constexpr float HALF_TRACK     = 0.11f;    
constexpr float K_ROT          = HALF_WHEELBASE + HALF_TRACK;  

// ---- Performance limits ----
constexpr float MAX_WHEEL_MPS  = 1.8f;     
constexpr float MAX_LIN_ACCEL  = 3.0f;     
constexpr float MAX_LIN_DECEL  = 4.0f;     
constexpr float MAX_ANG_ACCEL  = 12.0f;    

// ---- Control loop ----
constexpr uint32_t CONTROL_PERIOD_US = 5000;   
constexpr uint32_t CMD_TIMEOUT_MS    = 250;    
// ---- Position controller (target given in metres) ----
constexpr float POS_MAX_SPEED   = 1.8f;    // m/s cap while driving to a target
constexpr float POS_DEADBAND_M  = 0.02f;   // stop within this distance of the target
constexpr float POS_KP          = 6.0f;    // 1/s: velocity per metre of error near the target
constexpr float POS_BRAKE_FRAC  = 0.8f;    // use this fraction of MAX_LIN_DECEL for the braking curve
constexpr float POS_LIMIT_M     = 1.5f;    // reject targets farther than this from home
constexpr uint32_t ODO_PERIOD_MS = 50;     // telemetry period

constexpr float KP = 0.6f;                     
constexpr float KI = 4.0f;
constexpr float I_LIMIT = 0.3f;                
constexpr float VEL_FILTER_ALPHA = 0.4f;       

// ---- Encoders ----
constexpr float ENC_COUNTS_PER_WHEEL_REV = 660.0f;  

// ---- Pins ----
enum Wheel : uint8_t { FL = 0, FR = 1, RL = 2, RR = 3, NUM_WHEELS = 4 };

struct MotorPins { uint8_t pwm, in1, in2, ledcChannel; };
constexpr MotorPins MOTOR[NUM_WHEELS] = {
  /* FL */ {25, 26, 27, 0},
  /* FR */ {13, 32, 33, 1},
  /* RL */ {18, 19, 21, 2},
  /* RR */ {22, 23,  4, 3},
};

struct EncPins { uint8_t a, b; };
constexpr EncPins ENC[NUM_WHEELS] = {
  /* FL */ {34, 35},   
  /* FR */ {36, 39},
  /* RL */ {16, 17},
  /* RR */ {14,  5},
};

constexpr bool MOTOR_INVERT[NUM_WHEELS] = { false, true, false, true };
constexpr bool ENC_INVERT[NUM_WHEELS]   = { false, true, false, true };

// ============================== LEDC WRAPPER ================================

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  #define PWM_ATTACH(pin, ch)  ledcAttach((pin), PWM_FREQ_HZ, PWM_RES_BITS)
  #define PWM_WRITE(pin, ch, duty) ledcWrite((pin), (duty))
#else
  #define PWM_ATTACH(pin, ch)  do { ledcSetup((ch), PWM_FREQ_HZ, PWM_RES_BITS); ledcAttachPin((pin), (ch)); } while (0)
  #define PWM_WRITE(pin, ch, duty) ledcWrite((ch), (duty))
#endif

// ================================ STATE =====================================

static float tgtX = 0, tgtY = 0, tgtW = 0;   
static float curX = 0, curY = 0, curW = 0;   
static uint32_t lastCmdMs = 0;
static uint32_t lastCtrlUs = 0;

enum Mode : uint8_t { MODE_VELOCITY = 0, MODE_POSITION = 1 };
static Mode  mode = MODE_VELOCITY;
static float posX = 0, posY = 0;             // odometry (m), robot home frame
static float goalX = 0, goalY = 0;           // position target (m)
static float odoVX = 0, odoVY = 0;           // measured body velocity (m/s)
static uint32_t lastOdoMs = 0;

static float integ[NUM_WHEELS]    = {0};
static float measNorm[NUM_WHEELS] = {0};
static float wheelMps[NUM_WHEELS] = {0};     // unfiltered wheel speeds (m/s) for odometry

#if USE_ENCODERS
static volatile int32_t encCount[NUM_WHEELS] = {0};
static int32_t lastCount[NUM_WHEELS] = {0};

static inline void IRAM_ATTR encTick(uint8_t i) {
  int a = digitalRead(ENC[i].a);
  int b = digitalRead(ENC[i].b);
  int step = (a == b) ? 1 : -1;
  encCount[i] += ENC_INVERT[i] ? -step : step;
}
static void IRAM_ATTR encISR0() { encTick(0); }
static void IRAM_ATTR encISR1() { encTick(1); }
static void IRAM_ATTR encISR2() { encTick(2); }
static void IRAM_ATTR encISR3() { encTick(3); }
#endif

// ============================ LOW-LEVEL MOTOR ===============================

static void driveMotor(uint8_t w, float u) {
  u = constrain(u, -1.0f, 1.0f);
  if (MOTOR_INVERT[w]) u = -u;

  float mag = fabsf(u);
  uint32_t duty = 0;
  if (mag > 0.01f) {
    duty = (uint32_t)((MIN_DUTY + (1.0f - MIN_DUTY) * mag) * PWM_MAX);
    if (duty > PWM_MAX) duty = PWM_MAX;
    digitalWrite(MOTOR[w].in1, u > 0 ? HIGH : LOW);
    digitalWrite(MOTOR[w].in2, u > 0 ? LOW : HIGH);
  } else {
    digitalWrite(MOTOR[w].in1, LOW);   
    digitalWrite(MOTOR[w].in2, LOW);
  }
  PWM_WRITE(MOTOR[w].pwm, MOTOR[w].ledcChannel, duty);
}

// ============================ MECANUM KINEMATICS ============================

static void inverseKinematics(float x, float y, float w, float out[NUM_WHEELS]) {
  const float rot = K_ROT * w;
  out[FL] = y + x - rot;
  out[FR] = y - x + rot;
  out[RL] = y - x - rot;
  out[RR] = y + x + rot;
}

static void desaturate(float &x, float &y, float &w) {
  float v[NUM_WHEELS];
  inverseKinematics(x, y, w, v);
  float peak = 0;
  for (uint8_t i = 0; i < NUM_WHEELS; i++) peak = fmaxf(peak, fabsf(v[i]));
  if (peak > MAX_WHEEL_MPS) {
    const float s = MAX_WHEEL_MPS / peak;
    x *= s; y *= s; w *= s;
  }
}

// ========================= ACCELERATION LIMITER =============================

static void slewVec2(float &cx, float &cy, float tx, float ty, float dt) {
  const float curMag = hypotf(cx, cy);
  const float tgtMag = hypotf(tx, ty);
  const float aMax   = (tgtMag < curMag) ? MAX_LIN_DECEL : MAX_LIN_ACCEL;
  const float maxStep = aMax * dt;

  const float dx = tx - cx, dy = ty - cy;
  const float dist = hypotf(dx, dy);
  if (dist <= maxStep) { cx = tx; cy = ty; }
  else { const float s = maxStep / dist; cx += dx * s; cy += dy * s; }
}

static void slewScalar(float &c, float t, float maxRate, float dt) {
  const float maxStep = maxRate * dt;
  const float d = t - c;
  if (d > maxStep) c += maxStep;
  else if (d < -maxStep) c -= maxStep;
  else c = t;
}

// ============================ ODOMETRY ======================================
// Forward kinematics (inverse of inverseKinematics above). Heading is assumed fixed
// (w ~ 0) because the robot never rotates; with encoders disabled we integrate the
// commanded velocity instead.
static void updateOdometry(float dt) {
#if USE_ENCODERS
  const float vFL = measNorm[FL] * MAX_WHEEL_MPS;
  const float vFR = measNorm[FR] * MAX_WHEEL_MPS;
  const float vRL = measNorm[RL] * MAX_WHEEL_MPS;
  const float vRR = measNorm[RR] * MAX_WHEEL_MPS;
  odoVY = 0.25f * (vFL + vFR + vRL + vRR);
  odoVX = 0.25f * (vFL - vFR - vRL + vRR);
#else
  odoVX = curX; odoVY = curY;
#endif
  posX += odoVX * dt;
  posY += odoVY * dt;
}

// Turns "where am I vs where I want to be" into a body velocity command, with a
// braking curve v = sqrt(2*a*d) so the robot arrives without overshoot.
static void positionControl(float &vx, float &vy) {
  const float ex = goalX - posX, ey = goalY - posY;
  const float d = hypotf(ex, ey);
  if (d < POS_DEADBAND_M) { vx = 0; vy = 0; return; }
  const float vBrake = sqrtf(2.0f * POS_BRAKE_FRAC * MAX_LIN_DECEL * d);
  const float vP     = POS_KP * d;
  float v = fminf(fminf(vBrake, POS_MAX_SPEED), fmaxf(vP, 0.15f));
  if (d > 0.10f) v = fminf(vBrake, POS_MAX_SPEED);       // far away: full braking curve
  vx = ex / d * v;
  vy = ey / d * v;
}

// ============================ CONTROL STEP ==================================

static void controlStep(float dt) {
  float tx = tgtX, ty = tgtY, tw = tgtW;
  if (mode == MODE_POSITION) {
    positionControl(tx, ty);
    tw = 0;
    desaturate(tx, ty, tw);
  } else if (millis() - lastCmdMs > CMD_TIMEOUT_MS) {
    tx = ty = tw = 0;
  }

  slewVec2(curX, curY, tx, ty, dt);
  slewScalar(curW, tw, MAX_ANG_ACCEL, dt);

  float v[NUM_WHEELS];
  inverseKinematics(curX, curY, curW, v);

  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    const float targetNorm = v[i] / MAX_WHEEL_MPS;   
    float u = targetNorm;                            

#if USE_ENCODERS
    const int32_t c = encCount[i];
    const float cps = (float)(c - lastCount[i]) / dt;
    lastCount[i] = c;
    const float mps = cps / ENC_COUNTS_PER_WHEEL_REV * (2.0f * PI * WHEEL_RADIUS_M);
    wheelMps[i] = mps;
    measNorm[i] += VEL_FILTER_ALPHA * ((mps / MAX_WHEEL_MPS) - measNorm[i]);

    const float err = targetNorm - measNorm[i];
    if (fabsf(targetNorm) < 0.01f && fabsf(measNorm[i]) < 0.05f) {
      integ[i] = 0;                                  
    } else {
      integ[i] = constrain(integ[i] + KI * err * dt, -I_LIMIT, I_LIMIT);
    }
    u = targetNorm + KP * err + integ[i];
#endif

    driveMotor(i, u);
  }
  updateOdometry(dt);
}

// ================================ PUBLIC API ================================

void robotBegin() {
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pinMode(MOTOR[i].in1, OUTPUT);
    pinMode(MOTOR[i].in2, OUTPUT);
    digitalWrite(MOTOR[i].in1, LOW);
    digitalWrite(MOTOR[i].in2, LOW);
    PWM_ATTACH(MOTOR[i].pwm, MOTOR[i].ledcChannel);
    PWM_WRITE(MOTOR[i].pwm, MOTOR[i].ledcChannel, 0);
  }
#if USE_ENCODERS
  void (*isr[NUM_WHEELS])() = { encISR0, encISR1, encISR2, encISR3 };
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    pinMode(ENC[i].a, INPUT);
    pinMode(ENC[i].b, INPUT);
    attachInterrupt(digitalPinToInterrupt(ENC[i].a), isr[i], CHANGE);
  }
#endif
  lastCtrlUs = micros();
  lastCmdMs = millis();
}

void motionUpdate() {
  const uint32_t now = micros();
  const uint32_t dtUs = now - lastCtrlUs;
  if (dtUs < CONTROL_PERIOD_US) return;
  lastCtrlUs = now;
  controlStep(fminf(dtUs * 1e-6f, 0.05f));
}

void moveRobot(float x_velocity, float y_velocity, float rotation_speed) {
  mode = MODE_VELOCITY;
  tgtX = x_velocity; tgtY = y_velocity; tgtW = rotation_speed;
  desaturate(tgtX, tgtY, tgtW);
  lastCmdMs = millis();
  motionUpdate();
}

void stopRobot() { moveRobot(0, 0, 0); }

void goToPosition(float x, float y) {
  goalX = constrain(x, -POS_LIMIT_M, POS_LIMIT_M);
  goalY = constrain(y, -POS_LIMIT_M, POS_LIMIT_M);
  mode = MODE_POSITION;
  lastCmdMs = millis();
}

void holdHere() { goToPosition(posX, posY); }

void zeroOdometry() { posX = posY = 0; goalX = goalY = 0; }

// ================================ ARDUINO ===================================

// ---- Non-blocking line parser: never stalls the 200 Hz control loop ----
static char rxBuf[64];
static uint8_t rxLen = 0;

// Parses up to `max` comma separated floats from "a,b,c". Returns how many were read.
static int parseFloats(const char *p, float *out, int max) {
  int n = 0;
  while (*p && n < max) {
    char *end;
    float v = strtof(p, &end);
    if (end == p) break;
    out[n++] = v;
    p = end;
    while (*p == ' ' || *p == ',') p++;
  }
  return n;
}

static void handleLine(char *line) {
  // expected form: <WORD> or <WORD,arg,arg,...>
  if (line[0] != '<') return;
  char *close = strchr(line, '>');
  if (!close) return;
  *close = 0;
  char *body = line + 1;
  char *comma = strchr(body, ',');
  const char *args = "";
  if (comma) { *comma = 0; args = comma + 1; }

  float v[3];
  if (strcmp(body, "POS") == 0) {
    if (parseFloats(args, v, 2) == 2 && isfinite(v[0]) && isfinite(v[1])) {
      goToPosition(v[0], v[1]);
      Serial.printf("<ACK,POS,%.3f,%.3f>\n", goalX, goalY);
    } else {
      Serial.println("<ERR,POS>");
    }
  } else if (strcmp(body, "HOME") == 0) {
    goToPosition(0, 0);
    Serial.println("<ACK,HOME>");
  } else if (strcmp(body, "STOP") == 0) {
    holdHere();
    Serial.println("<ACK,STOP>");
  } else if (strcmp(body, "ZERO") == 0) {
    zeroOdometry();
    Serial.println("<ACK,ZERO>");
  } else if (strcmp(body, "VEL") == 0) {
    if (parseFloats(args, v, 3) == 3) { moveRobot(v[0], v[1], v[2]); Serial.println("<ACK,VEL>"); }
    else Serial.println("<ERR,VEL>");
  } else {
    Serial.println("<ERR,UNKNOWN>");
  }
}

static void serialPoll() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) { rxBuf[rxLen] = 0; handleLine(rxBuf); rxLen = 0; }
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxLen = 0;   // overflow: drop the garbage line
    }
  }
}

static void odoTelemetry() {
  const uint32_t now = millis();
  if (now - lastOdoMs >= ODO_PERIOD_MS) {
    lastOdoMs = now;
    Serial.printf("<ODO,%.3f,%.3f,%.3f,%.3f>\n", posX, posY, odoVX, odoVY);
  }
}

void setup() {
  Serial.begin(115200);
  robotBegin();
  goToPosition(0, 0);
}

void loop() {
  serialPoll();
  motionUpdate();      // 200 Hz PI wheel loops + position controller + odometry
  odoTelemetry();
}
