// Minimal 2-servo jog controller for the Waveshare ESP32 servo driver board.
//
// Replaces the stock ServoDriverST example, which was unreliable from a
// phone (OLED/RGB/ESP-NOW/dual-core background tasks all competing with the
// web server for CPU time) and required searching for + selecting an
// "active" servo before you could move it. This version does exactly one
// thing: the ESP32 hosts its own WiFi hotspot, your phone connects directly
// (no router needed) and opens a page with two fixed Left/Right button
// pairs, one per servo -- no search, no selection step. Buttons support
// press-and-hold (repeats while held, not just one step per tap). A small
// OLED (same one the stock firmware used) shows the AP info and both
// joints' current position. A "Set ID" field lets you permanently
// reassign a servo's bus ID (written to its EEPROM) without re-flashing.
//
// Wiring is unchanged from the stock firmware: servo bus on Serial1
// (RX=GPIO18, TX=GPIO19, 1,000,000 baud), OLED on I2C (SDA=GPIO21,
// SCL=GPIO22, SSD1306 128x32 @ 0x3C).

#include <WiFi.h>
#include <WebServer.h>
#include <SCServo.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>

// --- WiFi AP: phone connects directly to this, no router involved ---
const char *AP_SSID = "ArmServoCtrl";
const char *AP_PASSWORD = "12345678";  // ESP32 AP mode requires >=8 chars

// --- Servo bus (same wiring as the stock firmware) ---
#define SERVO_RXD 18
#define SERVO_TXD 19
#define SERVO_BAUD 1000000

// --- OLED (same wiring/panel as the stock firmware) ---
#define OLED_SDA 21
#define OLED_SCL 22
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 32
#define SCREEN_ADDRESS 0x3C

const int16_t JOG_STEP = 40;      // ticks per jog step (~3.5 deg @ 4096 ticks/rev)
const uint16_t JOG_SPEED = 800;

// Joint->servo-ID mapping is mutable (not const) so the "Set ID" feature
// can update it live when you rename the servo currently plugged into
// joint 1 or joint 2, without needing a re-flash.
uint8_t joint1Id = 1;
uint8_t joint2Id = 2;

SMS_STS st;
WebServer server(80);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

int16_t targetPos[2];  // [0]=joint1, [1]=joint2, in servo ticks (0-4095)
unsigned long lastScreenUpdate = 0;

uint8_t idFor(int joint) {
  return joint == 1 ? joint1Id : joint2Id;
}

void jog(int joint, int dir) {
  int idx = joint - 1;
  targetPos[idx] = constrain(targetPos[idx] + dir * JOG_STEP, 0, 4095);
  st.WritePosEx(idFor(joint), targetPos[idx], JOG_SPEED, 0);
}

String posReply() {
  return String(targetPos[0]) + "," + String(targetPos[1]);
}

void updateScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(AP_SSID);
  display.println(WiFi.softAPIP());
  display.print("J1(id");
  display.print(joint1Id);
  display.print("):");
  display.println(targetPos[0] * 360.0 / 4096.0, 1);
  display.print("J2(id");
  display.print(joint2Id);
  display.print("):");
  display.println(targetPos[1] * 360.0 / 4096.0, 1);
  display.display();
}

// Reassign a servo's own persistent bus ID (written to its EEPROM). If the
// servo being renamed is whichever one is currently mapped to joint1/joint2,
// that mapping follows it automatically -- no re-flash needed afterwards.
void setServoId(uint8_t fromId, uint8_t toId, String &resultOut) {
  int check = st.ReadPos(fromId);
  if (check < 0) {
    resultOut = "error: no response from id " + String(fromId);
    return;
  }
  st.unLockEprom(fromId);
  st.writeByte(fromId, SMS_STS_ID, toId);
  st.LockEprom(toId);

  if (joint1Id == fromId) joint1Id = toId;
  if (joint2Id == fromId) joint2Id = toId;
  int pos = st.ReadPos(toId);
  if (joint1Id == toId) targetPos[0] = (pos >= 0) ? pos : 2047;
  if (joint2Id == toId) targetPos[1] = (pos >= 0) ? pos : 2047;

  resultOut = "ok: id " + String(fromId) + " -> " + String(toId);
}

const char PAGE[] PROGMEM = R"HTML(
<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arm Servo Jog</title>
<style>
body{font-family:sans-serif;text-align:center;background:#111;color:#eee;
  -webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
h2{margin-top:1.4em}
button{font-size:2em;width:3.5em;height:2em;margin:0.3em;border-radius:0.3em;border:none;
  touch-action:none;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}
.pos{font-size:1.1em;color:#8f8}
input{width:3em;font-size:1.1em}
</style></head><body>
<h2>Joint 1</h2>
<button oncontextmenu="return false" onpointerdown="startJog(event,1,-1)" onpointerup="stopJog()" onpointerleave="stopJog()" onpointercancel="stopJog()">&#9664;</button>
<button oncontextmenu="return false" onpointerdown="startJog(event,1,1)" onpointerup="stopJog()" onpointerleave="stopJog()" onpointercancel="stopJog()">&#9654;</button>
<div class="pos" id="p1">-</div>
<h2>Joint 2</h2>
<button oncontextmenu="return false" onpointerdown="startJog(event,2,-1)" onpointerup="stopJog()" onpointerleave="stopJog()" onpointercancel="stopJog()">&#9664;</button>
<button oncontextmenu="return false" onpointerdown="startJog(event,2,1)" onpointerup="stopJog()" onpointerleave="stopJog()" onpointercancel="stopJog()">&#9654;</button>
<div class="pos" id="p2">-</div>
<h2>Servo ID</h2>
from <input type="number" id="fromId" value="1">
to <input type="number" id="toId" value="1">
<button style="font-size:1em;width:auto;padding:0 0.6em" onclick="setId()">Set ID</button>
<div class="pos" id="idmsg">-</div>
<script>
document.addEventListener('contextmenu', e => e.preventDefault());

function show(t){
  const parts = t.split(',');
  document.getElementById('p1').innerText = 'joint1: ' + parts[0];
  document.getElementById('p2').innerText = 'joint2: ' + parts[1];
}
function jog(j,d){ fetch('/jog?joint='+j+'&dir='+d).then(r=>r.text()).then(show); }

let holdTimer = null;
function startJog(ev,j,d){
  ev.preventDefault();
  jog(j,d);
  clearInterval(holdTimer);
  holdTimer = setInterval(()=>jog(j,d), 150);
}
function stopJog(){ clearInterval(holdTimer); holdTimer = null; }

function setId(){
  const f = document.getElementById('fromId').value;
  const t = document.getElementById('toId').value;
  fetch('/setid?from='+f+'&to='+t).then(r=>r.text()).then(m=>{
    document.getElementById('idmsg').innerText = m;
  });
}

setInterval(()=>{ fetch('/pos').then(r=>r.text()).then(show); }, 1000);
</script>
</body></html>
)HTML";

void handleRoot() {
  server.send(200, "text/html", PAGE);
}

void handleJog() {
  int joint = server.arg("joint").toInt();
  int dir = server.arg("dir").toInt();
  if (joint == 1 || joint == 2) {
    jog(joint, dir > 0 ? 1 : -1);
  }
  server.send(200, "text/plain", posReply());
}

void handlePos() {
  server.send(200, "text/plain", posReply());
}

void handleSetId() {
  int from = server.arg("from").toInt();
  int to = server.arg("to").toInt();
  if (from < 1 || from > 253 || to < 1 || to > 253) {
    server.send(400, "text/plain", "bad id (must be 1-253)");
    return;
  }
  String result;
  setServoId((uint8_t)from, (uint8_t)to, result);
  server.send(200, "text/plain", result);
}

void setup() {
  Serial.begin(115200);

  Serial1.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RXD, SERVO_TXD);
  st.pSerial = &Serial1;

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("SSD1306 init failed");
  }
  display.clearDisplay();
  display.display();

  // Seed targetPos from each servo's real current position so the first
  // jog step moves relative to where it physically is, not from zero.
  for (int j = 1; j <= 2; j++) {
    int pos = st.ReadPos(idFor(j));
    targetPos[j - 1] = (pos >= 0) ? pos : 2047;  // fall back to mid-range on read error
    st.EnableTorque(idFor(j), 1);
  }

  WiFi.softAP(AP_SSID, AP_PASSWORD);
  Serial.print("AP started: \"");
  Serial.print(AP_SSID);
  Serial.print("\"  password: \"");
  Serial.print(AP_PASSWORD);
  Serial.println("\"");
  Serial.print("Connect your phone, then browse to http://");
  Serial.println(WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/jog", handleJog);
  server.on("/pos", handlePos);
  server.on("/setid", handleSetId);
  server.begin();

  updateScreen();
}

void loop() {
  server.handleClient();
  if (millis() - lastScreenUpdate > 400) {
    updateScreen();
    lastScreenUpdate = millis();
  }
}
