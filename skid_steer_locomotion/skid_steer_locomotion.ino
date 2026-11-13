/*
 * Autonomous Trash Catching Robot - Skid-steer drive + WiFi joystick (ESP32 + 2x BTS7960)
 *
 * WiFi: the ESP32 makes its own network. Connect your phone to it, open
 * http://192.168.4.1 in the browser and drive with the on-screen joystick.
 * PC LINK: the computer (OpenCV / trajectory code) talks to the car over WiFi (UDP port 4210)
 * with the same text commands as the serial firmware:
 *     <VEL,vx,vy,w>  <POS,x,y>  <HOME>  <STOP>  <ZERO>  <PING>
 *   x right [m], y forward [m] (relative to the ZERO point), w [rad/s] CCW. vx is ignored.
 *   Replies: <ACK,CMD> and <ODO,x,y,vx,vy> every 50 ms to whoever sent the last command.
 *   Position control: if the target is near the car's own driving line it just drives
 *   forward/back; otherwise it turns toward it (reverse allowed) and drives.
 * The phone arrow-button page still works for manual driving.
 * FINAL: arrow-button page (hold to move, STOP in the middle) with live wheel speeds,
 *        calibrated constants, side trim. Use THIS file only.
 * Also: control loop runs in its own task (a slow WiFi client can no longer freeze the
 *     motors), optional router (STA) mode, and a UDP port for a computer / OpenCV.
 *
 * Works with ESP32 Arduino core 2.x and 3.x (ledc API differences handled below).
 *
 * Body frame / sign convention:
 *   y_velocity     [m/s]   + = FORWARD
 *   rotation_speed [rad/s] + = counter-clockwise (viewed from above)
 *   x_velocity is accepted but IGNORED (normal wheels cannot strafe).
 *
 * Hardware:
 *   LEFT  BTS7960 drives FL + RL motors in parallel   RPWM=25  LPWM=26
 *   RIGHT BTS7960 drives FR + RR motors in parallel   RPWM=32  LPWM=33
 *   BTS VCC, R_EN, L_EN -> 3V3.  R_IS / L_IS unconnected.
 *   Encoders (A / B): FL 34/35, FR 36/39, RL 16/17, RR 14/5
 *
 * Control: slew limiter -> side kinematics -> PI per side (average of the two
 * encoders on that side) -> PWM.
 *
 * FIRST TIME: set USE_ENCODERS 0, wheels off the ground, and drive from the phone.
 * Then use ENCODER_TEST 1 to check the encoder signs before closing the loop.
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <WebServer.h>

// ============================ CONFIGURATION =================================

#define USE_ENCODERS 1      // 1 = closed-loop speed PI, 0 = open loop
#define ENCODER_TEST 0      // 1 = motors off, print raw encoder counts (needs USE_ENCODERS 1)
#define DEBUG_PRINT  0      // 1 = print target / measured side speeds
#define PAIRING_TEST 0      // 1 = drive LEFT then RIGHT board alone, print which encoders respond (needs USE_ENCODERS 1)

#if !USE_ENCODERS
  #error "PC link (odometry / position control) needs USE_ENCODERS 1"
#endif
#if ENCODER_TEST && !USE_ENCODERS
  #error "ENCODER_TEST needs USE_ENCODERS 1"
#endif

// ---- PWM (ledc) ----
constexpr uint32_t PWM_FREQ_HZ  = 20000;
constexpr uint8_t  PWM_RES_BITS = 10;
constexpr uint32_t PWM_MAX      = (1u << PWM_RES_BITS) - 1;
constexpr float    MIN_DUTY     = 0.08f;   // duty to overcome stiction (tune)
constexpr float    MAX_DUTY     = 1.0f;    // lower (e.g. 0.8) to cap motor current

// ---- Geometry ----
constexpr float WHEEL_RADIUS_M = 0.0425f;  // 85 mm wheel
// Half the distance between the left and right wheel lines. Skid steer scrubs, so
// the effective value is larger than the real one; tune until a commanded spin
// turns the robot by the expected angle.
constexpr float HALF_TRACK     = 0.23f;   // calibrated from spin test (geometric 0.127; skid steer scrubs)

// ---- Performance limits ----
constexpr float MAX_WHEEL_MPS  = 1.2f;     // wheel surface speed at full duty, loaded
constexpr float MAX_LIN_ACCEL  = 5.0f;     // m/s^2 (about 0.25 s to full speed)
constexpr float MAX_LIN_DECEL  = 6.0f;     // m/s^2
constexpr float MAX_ANG_ACCEL  = 20.0f;    // rad/s^2

// ---- Control loop ----
constexpr uint32_t CONTROL_PERIOD_US = 5000;   // 200 Hz
constexpr uint32_t CMD_TIMEOUT_MS    = 150;    // no command for this long -> ramp to stop

// ---- Position control (<POS,x,y>) ----
constexpr float    POS_TOL_M       = 0.02f;   // arrived when closer than this
constexpr float    LAT_TOL_M       = 0.06f;   // target this close to the car's own line -> just drive straight
constexpr float    POS_MAX_V       = 0.9f;    // m/s while going to a point
constexpr float    POS_MAX_W       = 3.0f;    // rad/s while turning toward a point
constexpr float    K_HEAD          = 4.0f;    // turn rate per radian of heading error
constexpr float    HEAD_DRIVE_RAD  = 0.35f;   // may drive while heading error is below this (~20 deg)
constexpr uint32_t POS_TIMEOUT_MS  = 3000;    // no new <POS> for this long -> stop
constexpr float KP = 0.6f;
constexpr float KI = 4.0f;
constexpr float I_LIMIT = 0.3f;
constexpr float VEL_FILTER_ALPHA = 0.4f;

// ---- Encoders ----
// JGB37-520 listing: 20 PPR. Counts = PPR * 2 (A edge both ways) * gear ratio.
// 20 * 2 * 18 = 720 assumes an 18:1 gearbox. CALIBRATE: turn one wheel exactly one
// revolution by hand with ENCODER_TEST 1 and use the number it prints.
constexpr float ENC_COUNTS_PER_WHEEL_REV = 660.0f;   // confirmed by hand (30:1, 11 PPR x 2 edges)

// ---- Pins ----
enum Side  : uint8_t { LEFT = 0, RIGHT = 1, NUM_SIDES = 2 };
enum Wheel : uint8_t { FL = 0, FR = 1, RL = 2, RR = 3, NUM_WHEELS = 4 };

struct SidePins { uint8_t rpwm, lpwm, chR, chL; };
constexpr SidePins SIDE[NUM_SIDES] = {
  /* LEFT  */ {32, 33, 2, 3},   // this board drives the FL + RL motors (pairing test)
  /* RIGHT */ {25, 26, 0, 1},   // this board drives the FR + RR motors
};

struct EncPins { uint8_t a, b; };
constexpr EncPins ENC[NUM_WHEELS] = {
  /* FL */ {34, 35},
  /* FR */ {36, 39},
  /* RL */ {16, 17},
  /* RR */ {14,  5},
};

// Flip if a side spins the wrong way for a positive command (mirrored mounting)
constexpr bool MOTOR_INVERT[NUM_SIDES]  = { false, true };

// Straight-line trim. Car veers LEFT -> lower RIGHT (e.g. 0.98). Car veers RIGHT -> lower LEFT.
constexpr float SIDE_TRIM[NUM_SIDES] = { 1.00f, 0.98f };   // LEFT, RIGHT
// Flip if a wheel's encoder count goes DOWN when that wheel turns "forward" by hand
constexpr bool ENC_INVERT[NUM_WHEELS]   = { false, true, false, true };

// ============================== LEDC WRAPPER ================================

#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  #define PWM_ATTACH(pin, ch)      ledcAttach((pin), PWM_FREQ_HZ, PWM_RES_BITS)
  #define PWM_WRITE(pin, ch, duty) ledcWrite((pin), (duty))
#else
  #define PWM_ATTACH(pin, ch)      do { ledcSetup((ch), PWM_FREQ_HZ, PWM_RES_BITS); ledcAttachPin((pin), (ch)); } while (0)
  #define PWM_WRITE(pin, ch, duty) ledcWrite((ch), (duty))
#endif

// ================================ STATE =====================================

static float tgtY = 0, tgtW = 0;     // requested chassis velocity (desaturated)
static float curY = 0, curW = 0;     // slew-limited velocity actually applied
static uint32_t lastCmdMs  = 0;
static uint32_t lastCtrlUs = 0;

enum Mode : uint8_t { MODE_VEL = 0, MODE_POS = 1 };
static volatile Mode mode = MODE_VEL;
static float odoX = 0, odoY = 0, odoTh = 0;     // m, m, rad (theta CCW from +Y)
static float odoVx = 0, odoVy = 0;
static float posTx = 0, posTy = 0;
static uint32_t lastPosMs = 0;

static float integ[NUM_SIDES]    = {0};
static float measNorm[NUM_SIDES] = {0};
static float dbgTarget[NUM_SIDES] = {0};

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

// GPIO 34-39 are input-only and have no internal pull-ups
static void encPinMode(uint8_t p) {
  pinMode(p, (p >= 34 && p <= 39) ? INPUT : INPUT_PULLUP);
}
#endif

// ============================ LOW-LEVEL MOTOR ===============================

// u in [-1, 1]. Forward drives RPWM, reverse drives LPWM.
// u == 0 -> both PWM at 0 with EN high = motor terminals shorted to GND = BRAKE.
static void driveSide(uint8_t s, float u) {
  u = constrain(u, -1.0f, 1.0f);
  if (MOTOR_INVERT[s]) u = -u;

  const float mag = fabsf(u);
  uint32_t duty = 0;
  if (mag > 0.01f) {
    float d = MIN_DUTY + (1.0f - MIN_DUTY) * mag;
    if (d > MAX_DUTY) d = MAX_DUTY;
    duty = (uint32_t)(d * PWM_MAX);
  }

  if (u > 0) {
    PWM_WRITE(SIDE[s].lpwm, SIDE[s].chL, 0);
    PWM_WRITE(SIDE[s].rpwm, SIDE[s].chR, duty);
  } else {
    PWM_WRITE(SIDE[s].rpwm, SIDE[s].chR, 0);
    PWM_WRITE(SIDE[s].lpwm, SIDE[s].chL, duty);
  }
}

// ============================ SKID-STEER KINEMATICS =========================

// Chassis velocity -> left/right wheel surface speeds [m/s]
static void inverseKinematics(float y, float w, float out[NUM_SIDES]) {
  out[LEFT]  = y - HALF_TRACK * w;
  out[RIGHT] = y + HALF_TRACK * w;
}

// Scale the command so neither side exceeds MAX_WHEEL_MPS (keeps the turn ratio)
static void desaturate(float &y, float &w) {
  float v[NUM_SIDES];
  inverseKinematics(y, w, v);
  const float peak = fmaxf(fabsf(v[LEFT]), fabsf(v[RIGHT]));
  if (peak > MAX_WHEEL_MPS) {
    const float s = MAX_WHEEL_MPS / peak;
    y *= s; w *= s;
  }
}

// ========================= ACCELERATION LIMITER =============================

static void slewScalar(float &c, float t, float maxRate, float dt) {
  const float maxStep = maxRate * dt;
  const float d = t - c;
  if (d > maxStep) c += maxStep;
  else if (d < -maxStep) c -= maxStep;
  else c = t;
}

// ============================ CONTROL STEP ==================================

static float wrapPi(float a) {
  while (a >  PI) a -= 2.0f * PI;
  while (a < -PI) a += 2.0f * PI;
  return a;
}

// Sets tgtY / tgtW to move toward (posTx, posTy). Forward = (-sin th, cos th).
static void positionStep() {
  const float dx = posTx - odoX, dy = posTy - odoY;
  const float c = cosf(odoTh), s = sinf(odoTh);
  const float proj = -dx * s + dy * c;     // distance along the car's forward axis
  const float lat  =  dx * c + dy * s;     // distance along the car's right axis
  const float dist = sqrtf(dx * dx + dy * dy);

  if (dist < POS_TOL_M) { tgtY = 0; tgtW = 0; return; }

  const float aBrake = 0.6f * MAX_LIN_DECEL;
  if (fabsf(lat) < LAT_TOL_M || dist < 0.10f) {
    // On the car's line (or very close): drive straight, forward or reverse
    if (fabsf(proj) < POS_TOL_M) { tgtY = 0; tgtW = 0; return; }
    const float v = fminf(POS_MAX_V, sqrtf(2.0f * aBrake * fabsf(proj)));
    tgtY = (proj >= 0) ? v : -v;
    tgtW = 0;
    return;
  }

  // Off the line: turn toward the target (reverse allowed), then drive
  const float bearing = atan2f(-dx, dy);
  float err = wrapPi(bearing - odoTh);
  float dir = 1.0f;
  if (fabsf(err) > PI / 2) { err = wrapPi(err + PI); dir = -1.0f; }
  tgtW = constrain(K_HEAD * err, -POS_MAX_W, POS_MAX_W);
  if (fabsf(err) > HEAD_DRIVE_RAD) tgtY = 0;
  else tgtY = dir * fminf(POS_MAX_V, sqrtf(2.0f * aBrake * dist)) * (1.0f - fabsf(err) / HEAD_DRIVE_RAD);
}

static void controlStep(float dt) {
  // 0) Measure wheels + integrate odometry (exact distances from the encoder counts)
  float wheelDist[NUM_WHEELS];
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    const int32_t c = encCount[i];
    wheelDist[i] = (float)(c - lastCount[i]) / ENC_COUNTS_PER_WHEEL_REV * (2.0f * PI * WHEEL_RADIUS_M);
    lastCount[i] = c;
  }
  const float dL = 0.5f * (wheelDist[FL] + wheelDist[RL]);
  const float dR = 0.5f * (wheelDist[FR] + wheelDist[RR]);
  const float dS = 0.5f * (dL + dR);
  const float dTh = (dR - dL) / (2.0f * HALF_TRACK);
  const float thMid = odoTh + 0.5f * dTh;
  odoX  += -dS * sinf(thMid);
  odoY  +=  dS * cosf(thMid);
  odoTh += dTh;
  const float sideMps[NUM_SIDES] = { dL / dt, dR / dt };

  // 1) Position mode computes the velocity target itself
  if (mode == MODE_POS) {
    if (millis() - lastPosMs > POS_TIMEOUT_MS) {
      mode = MODE_VEL; tgtY = tgtW = 0;
    } else {
      positionStep();
      desaturate(tgtY, tgtW);
    }
    lastCmdMs = millis();                  // position mode is not subject to the 150 ms failsafe
  }

  // Failsafe: stale velocity command -> target zero (still decelerates through the limiter)
  float ty = tgtY, tw = tgtW;
  if (millis() - lastCmdMs > CMD_TIMEOUT_MS) { ty = tw = 0; }

  // 2) Acceleration limiting on the chassis velocity
  slewScalar(curY, ty, (fabsf(ty) < fabsf(curY)) ? MAX_LIN_DECEL : MAX_LIN_ACCEL, dt);
  slewScalar(curW, tw, MAX_ANG_ACCEL, dt);

  // 3) Kinematics -> side speeds
  float v[NUM_SIDES];
  inverseKinematics(curY, curW, v);
  v[LEFT]  *= SIDE_TRIM[LEFT];
  v[RIGHT] *= SIDE_TRIM[RIGHT];

  // 4) Per-side speed control + PWM
  for (uint8_t s = 0; s < NUM_SIDES; s++) {
    const float targetNorm = v[s] / MAX_WHEEL_MPS;   // -1..1
    dbgTarget[s] = v[s];

    measNorm[s] += VEL_FILTER_ALPHA * ((sideMps[s] / MAX_WHEEL_MPS) - measNorm[s]);

    const float err = targetNorm - measNorm[s];
    if (fabsf(targetNorm) < 0.01f && fabsf(measNorm[s]) < 0.05f) {
      integ[s] = 0;                                  // at rest: no residual drive
    } else {
      integ[s] = constrain(integ[s] + KI * err * dt, -I_LIMIT, I_LIMIT);
    }
    driveSide(s, targetNorm + KP * err + integ[s]);
  }

  // 5) Filtered world-frame velocity for <ODO>
  const float vf = 0.5f * (measNorm[LEFT] + measNorm[RIGHT]) * MAX_WHEEL_MPS;
  odoVx = -vf * sinf(odoTh);
  odoVy =  vf * cosf(odoTh);
}

// ================================ PUBLIC API ================================

void robotBegin() {
  for (uint8_t s = 0; s < NUM_SIDES; s++) {
    PWM_ATTACH(SIDE[s].rpwm, SIDE[s].chR);
    PWM_ATTACH(SIDE[s].lpwm, SIDE[s].chL);
    PWM_WRITE(SIDE[s].rpwm, SIDE[s].chR, 0);
    PWM_WRITE(SIDE[s].lpwm, SIDE[s].chL, 0);
  }
#if USE_ENCODERS
  void (*isr[NUM_WHEELS])() = { encISR0, encISR1, encISR2, encISR3 };
  for (uint8_t i = 0; i < NUM_WHEELS; i++) {
    encPinMode(ENC[i].a);
    encPinMode(ENC[i].b);
    attachInterrupt(digitalPinToInterrupt(ENC[i].a), isr[i], CHANGE);
  }
#endif
  lastCtrlUs = micros();
  lastCmdMs  = millis();
}

// Runs one control step if the fixed period has elapsed. Safe to call every loop().
void motionUpdate() {
  const uint32_t now = micros();
  const uint32_t dtUs = now - lastCtrlUs;
  if (dtUs < CONTROL_PERIOD_US) return;
  lastCtrlUs = now;
  controlStep(fminf(dtUs * 1e-6f, 0.05f));
}

/**
 * Command the robot.
 *   x_velocity     ignored (kept so existing code compiles)
 *   y_velocity     [m/s]   + = forward
 *   rotation_speed [rad/s] + = counter-clockwise
 */
void moveRobot(float x_velocity, float y_velocity, float rotation_speed) {
  (void)x_velocity;
  tgtY = y_velocity; tgtW = rotation_speed;
  desaturate(tgtY, tgtW);
  lastCmdMs = millis();
}

void stopRobot() { moveRobot(0, 0, 0); }

// ================================ WIFI ======================================

// Router mode: fill in your 2.4 GHz WiFi name + password (the ESP32 has NO 5 GHz).
// Leave STA_SSID empty to make the ESP32 create its own network instead.
constexpr char STA_SSID[] = "";
constexpr char STA_PASS[] = "";
constexpr uint16_t UDP_PORT = 4210;        // computer sends "y,w" here (both -1..1)

constexpr char AP_SSID[] = "TrashBot";
constexpr char AP_PASS[] = "12345678";     // at least 8 characters; change it
constexpr float WIFI_MAX_Y = 1.0f;         // joystick full forward  [m/s]
constexpr float WIFI_MAX_W = 3.0f;         // full turn button [rad/s] at 100% speed

static WebServer server(80);

static const char PAGE[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center;touch-action:none;user-select:none;-webkit-user-select:none}
.pad{display:grid;grid-template-columns:repeat(3,92px);grid-template-rows:repeat(3,92px);gap:10px;justify-content:center;margin:18px auto}
.b{background:#333;border-radius:16px;font-size:34px;display:flex;align-items:center;justify-content:center;touch-action:none}
.b.on{background:#4a9}
</style></head><body>
<h3>TrashBot</h3>
<div class="pad">
<div class="b" id="bu" style="grid-column:2;grid-row:1">&#9650;&#xFE0E;</div>
<div class="b" id="bl" style="grid-column:1;grid-row:2">&#9664;&#xFE0E;</div>
<div class="b" id="bs" style="grid-column:2;grid-row:2;background:#a33;font-size:20px">STOP</div>
<div class="b" id="br" style="grid-column:3;grid-row:2">&#9654;&#xFE0E;</div>
<div class="b" id="bd" style="grid-column:2;grid-row:3">&#9660;&#xFE0E;</div>
</div>
Speed <input type="range" id="spd" min="20" max="100" value="60"> <span id="sv">60%</span>
<div id="st">ready</div>
<div id="info" style="font-size:14px;margin-top:8px">tap here to reset peak</div>
<script>
const pressed=new Set();
let busy=false, peak=0;
const spd=document.getElementById('spd'), sv=document.getElementById('sv'),
      st=document.getElementById('st'), info=document.getElementById('info');
spd.oninput=()=>{sv.textContent=spd.value+'%';};
function req(url){
  const ctl=new AbortController();
  const to=setTimeout(()=>ctl.abort(),250);
  return fetch(url,{signal:ctl.signal,cache:'no-store'}).finally(()=>clearTimeout(to));
}
function send(){
  if(busy)return; busy=true;
  let fy=0,fx=0;
  if(pressed.has('u'))fy+=1; if(pressed.has('d'))fy-=1;
  if(pressed.has('r'))fx+=1; if(pressed.has('l'))fx-=1;
  const s=spd.value/100;
  req('/drive?y='+(fy*s).toFixed(2)+'&w='+(-fx*s).toFixed(2))
    .then(()=>{st.textContent='connected';})
    .catch(()=>{st.textContent='no link';})
    .finally(()=>{busy=false;});
}
function flush(){ if(busy){setTimeout(flush,20);} else send(); }
[['bu','u'],['bd','d'],['bl','l'],['br','r']].forEach(([id,k])=>{
  const el=document.getElementById(id);
  el.addEventListener('pointerdown',e=>{e.preventDefault();el.setPointerCapture(e.pointerId);pressed.add(k);el.classList.add('on');send();});
  const up=()=>{pressed.delete(k);el.classList.remove('on');flush();};
  el.addEventListener('pointerup',up);
  el.addEventListener('pointercancel',up);
  el.addEventListener('lostpointercapture',up);
});
function stopAll(){
  pressed.clear();
  document.querySelectorAll('.b').forEach(x=>x.classList.remove('on'));
  flush();
}
document.getElementById('bs').addEventListener('pointerdown',stopAll);
window.addEventListener('blur',stopAll);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopAll();});
setInterval(()=>{if(pressed.size)send();},60);

info.onclick=()=>{peak=0;};
setInterval(()=>{
  req('/status').then(r=>r.text()).then(t=>{
    const a=t.split(',').map(Number);
    const v=(Math.abs(a[1])+Math.abs(a[3]))/2;
    if(v>peak)peak=v;
    info.textContent='L '+a[1].toFixed(2)+'  R '+a[3].toFixed(2)+'  peak '+peak.toFixed(2)+' m/s';
  }).catch(()=>{});
},500);
</script></body></html>
)rawliteral";

static void handleRoot()  { server.send_P(200, "text/html", PAGE); }

static WiFiUDP udp;

// y, w normalised to -1..1  (w + = counter-clockwise)
static void applyDrive(float y, float w) {
  y = constrain(y, -1.0f, 1.0f) * WIFI_MAX_Y;
  w = constrain(w, -1.0f, 1.0f) * WIFI_MAX_W;
  moveRobot(0.0f, y, w);
}

static IPAddress rIP;
static uint16_t  rPort = 0;

static void sendTo(const char *msg) {
  if (!rPort) return;
  udp.beginPacket(rIP, rPort);
  udp.write((const uint8_t *)msg, strlen(msg));
  udp.endPacket();
}

// body = text between '<' and '>' e.g. "POS,0.30,0.00"
static void handleCmd(char *body) {
  char *cmd = strtok(body, ",");
  if (!cmd) return;
  float a[3] = {0, 0, 0};
  int n = 0;
  char *t;
  while (n < 3 && (t = strtok(nullptr, ","))) a[n++] = (float)atof(t);

  if (!strcmp(cmd, "VEL")) {                 // <VEL,vx,vy,w>   vx ignored (no strafing)
    mode = MODE_VEL;
    moveRobot(0.0f, a[1], a[2]);
  } else if (!strcmp(cmd, "POS")) {          // <POS,x,y> relative to ZERO point
    posTx = a[0]; posTy = a[1];
    lastPosMs = millis();
    mode = MODE_POS;
  } else if (!strcmp(cmd, "HOME")) {
    posTx = 0; posTy = 0;
    lastPosMs = millis();
    mode = MODE_POS;
  } else if (!strcmp(cmd, "STOP")) {
    mode = MODE_VEL;
    moveRobot(0.0f, 0.0f, 0.0f);
  } else if (!strcmp(cmd, "ZERO")) {
    odoX = 0; odoY = 0; odoTh = 0;
  } else if (strcmp(cmd, "PING") != 0) {
    return;                                  // unknown command: ignore
  }
  char reply[40];
  snprintf(reply, sizeof(reply), "<ACK,%s>\n", cmd);
  sendTo(reply);
}

static void pollUdp() {
  int n = udp.parsePacket();
  while (n > 0) {
    char buf[96];
    const int len = udp.read(buf, sizeof(buf) - 1);
    if (len <= 0) break;
    buf[len] = 0;
    rIP = udp.remoteIP();
    rPort = udp.remotePort();
    if (strchr(buf, '<')) {
      char *p = buf;
      while ((p = strchr(p, '<')) != nullptr) {
        char *e = strchr(p, '>');
        if (!e) break;
        *e = 0;
        handleCmd(p + 1);
        p = e + 1;
      }
    } else {                                 // legacy "y,w" (normalised -1..1)
      float y, w;
      if (sscanf(buf, "%f,%f", &y, &w) == 2) { mode = MODE_VEL; applyDrive(y, w); }
    }
    n = udp.parsePacket();
  }
}

// Motor control runs in its own task so a stuck WiFi client cannot stall it
static void controlTask(void *) {
  for (;;) {
    motionUpdate();
    vTaskDelay(1);
  }
}

static void handleStatus() {
  char b[80];
  snprintf(b, sizeof(b), "%.2f,%.2f,%.2f,%.2f",
           dbgTarget[LEFT],  measNorm[LEFT]  * MAX_WHEEL_MPS,
           dbgTarget[RIGHT], measNorm[RIGHT] * MAX_WHEEL_MPS);
  server.send(200, "text/plain", b);
}

static void handleDrive() {
  const float y = constrain(server.arg("y").toFloat(), -1.0f, 1.0f) * WIFI_MAX_Y;
  const float w = constrain(server.arg("w").toFloat(), -1.0f, 1.0f) * WIFI_MAX_W;
  moveRobot(0.0f, y, w);
  server.send(200, "text/plain", "ok");
}

// ================================ ARDUINO ===================================

void setup() {
  Serial.begin(115200);
  robotBegin();

  bool sta = false;
  if (STA_SSID[0]) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(STA_SSID, STA_PASS);
    const uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000) delay(200);
    sta = (WiFi.status() == WL_CONNECTED);
  }
  if (!sta) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
  }
  WiFi.setSleep(false);                 // WiFi power-save adds random lag

  server.on("/", handleRoot);
  server.on("/drive", handleDrive);
  server.on("/status", handleStatus);
  server.begin();
  udp.begin(UDP_PORT);

  Serial.print(sta ? "Router mode, open http://" : "Own network 'TrashBot', open http://");
  Serial.println(sta ? WiFi.localIP() : WiFi.softAPIP());
  Serial.print("UDP port ");  Serial.println(UDP_PORT);

#if !PAIRING_TEST && !ENCODER_TEST
  xTaskCreatePinnedToCore(controlTask, "ctl", 4096, nullptr, 2, nullptr, 1);
#endif
}

void loop() {
#if PAIRING_TEST
  // Wheels OFF the ground. Drives one board at 40% for 1.5 s, prints encoder deltas, repeats.
  static uint8_t phase = 0;
  static uint32_t tStart = 0;
  static int32_t base[NUM_WHEELS] = {0};
  if (tStart == 0 || millis() - tStart > 1500) {
    if (tStart != 0) {
      Serial.printf("%s board driven -> FL %ld  FR %ld  RL %ld  RR %ld\n",
                    phase == 0 ? "LEFT(25/26)" : "RIGHT(32/33)",
                    (long)(encCount[FL] - base[FL]), (long)(encCount[FR] - base[FR]),
                    (long)(encCount[RL] - base[RL]), (long)(encCount[RR] - base[RR]));
      driveSide(LEFT, 0); driveSide(RIGHT, 0);
      phase ^= 1;
      delay(800);
    }
    for (uint8_t i = 0; i < NUM_WHEELS; i++) base[i] = encCount[i];
    tStart = millis();
    driveSide(phase == 0 ? LEFT : RIGHT, 0.4f);
  }
  return;
#endif
#if ENCODER_TEST
  static uint32_t t0 = 0;
  if (millis() - t0 >= 200) {
    t0 = millis();
    Serial.printf("FL %ld  FR %ld  RL %ld  RR %ld\n",
                  (long)encCount[FL], (long)encCount[FR],
                  (long)encCount[RL], (long)encCount[RR]);
  }
  return;
#endif

  server.handleClient();   // phone joystick commands -> moveRobot()
  pollUdp();               // computer / OpenCV commands
  {
    static uint32_t tOdo = 0;
    if (millis() - tOdo >= 50) {
      tOdo = millis();
      char b[80];
      snprintf(b, sizeof(b), "<ODO,%.3f,%.3f,%.3f,%.3f>\n", odoX, odoY, odoVx, odoVy);
      sendTo(b);
    }
  }
  // motionUpdate() runs in controlTask; no command for CMD_TIMEOUT_MS -> ramps to stop

#if DEBUG_PRINT
  static uint32_t tp = 0;
  if (millis() - tp >= 200) {
    tp = millis();
    Serial.printf("L tgt %.2f meas %.2f | R tgt %.2f meas %.2f m/s\n",
                  dbgTarget[LEFT],  measNorm[LEFT]  * MAX_WHEEL_MPS,
                  dbgTarget[RIGHT], measNorm[RIGHT] * MAX_WHEEL_MPS);
  }
#endif
}
