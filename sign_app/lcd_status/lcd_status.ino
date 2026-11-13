// Grove LCD RGB Backlight (v5) status display for the sign program.
// The Linux side connects to 127.0.0.1:7500 (the router's monitor port) and sends one line per update:
//     T<line 1>\t<line 2>\t<r> <g> <b>\n      show two 16-character lines and set the backlight colour
//     ?\n                                      reply with "LCD text=<0|1> rgb=<address or 0>" (I2C devices that answered)
#include <Arduino_RouterBridge.h>
#include <Wire.h>

const uint8_t LCD = 0x3E;            // text controller
uint8_t rgbAddr = 0;                 // 0x30 on v5 boards, 0x62 on older ones, 0 if none answered
bool lcdOk = false;
String line;

bool present(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

void cmd(uint8_t c) {
  Wire.beginTransmission(LCD); Wire.write(0x80); Wire.write(c); Wire.endTransmission();
}

void rgbReg(uint8_t reg, uint8_t v) {
  Wire.beginTransmission(rgbAddr); Wire.write(reg); Wire.write(v); Wire.endTransmission();
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  if (rgbAddr == 0x30) { rgbReg(0x06, r); rgbReg(0x07, g); rgbReg(0x08, b); }
  else if (rgbAddr == 0x62) { rgbReg(0x04, r); rgbReg(0x03, g); rgbReg(0x02, b); }
}

void printLine(uint8_t row, const String &s) {
  cmd(0x80 | (row ? 0x40 : 0x00));
  for (uint8_t i = 0; i < 16; i++) {          // one character per transmission: sent together, every second one is lost
    Wire.beginTransmission(LCD); Wire.write(0x40); Wire.write(i < s.length() ? s[i] : ' '); Wire.endTransmission();
  }
}

void lcdInit() {
  lcdOk = present(LCD);
  rgbAddr = present(0x30) ? 0x30 : (present(0x62) ? 0x62 : 0);
  if (!lcdOk) return;
  delay(50);
  cmd(0x38); delay(5); cmd(0x38); delay(1); cmd(0x38);   // 2 lines, 5x8 font
  cmd(0x0C);                                             // display on, no cursor
  cmd(0x01); delay(2);                                   // clear
  cmd(0x06);                                             // left to right
  if (rgbAddr == 0x30) { rgbReg(0x00, 0x07); delay(1); rgbReg(0x04, 0x15); }
  else if (rgbAddr == 0x62) { rgbReg(0x00, 0x00); rgbReg(0x01, 0x00); rgbReg(0x08, 0xAA); }
  setRGB(60, 60, 60);
  printLine(0, "Sign program");
  printLine(1, "waiting...");
}

void handle(const String &s) {
  if (s == "?") {
    if (!lcdOk) lcdInit();                               // lets a display plugged in later come up
    Monitor.print("LCD text="); Monitor.print(lcdOk ? 1 : 0);
    Monitor.print(" rgb="); Monitor.println(rgbAddr);
    return;
  }
  if (s.length() < 1 || s[0] != 'T' || !lcdOk) return;
  int a = s.indexOf('\t'), b = a < 0 ? -1 : s.indexOf('\t', a + 1);
  if (a < 0 || b < 0) return;
  printLine(0, s.substring(1, a));
  printLine(1, s.substring(a + 1, b));
  int r = 0, g = 0, bl = 0;
  if (sscanf(s.substring(b + 1).c_str(), "%d %d %d", &r, &g, &bl) == 3) setRGB(r, g, bl);
}

void setup() {
  Bridge.begin();
  Monitor.begin(115200);
  Wire.begin();
  lcdInit();
}

void loop() {
  while (Monitor.available()) {
    char c = Monitor.read();
    if (c == '\n') { handle(line); line = ""; }
    else if (c != '\r' && line.length() < 80) line += c;
  }
  delay(5);
}
