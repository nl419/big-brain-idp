#include <robot_server.h>
// Irritatingly, for vscode to play nice, must include headers in both .h and .cpp files
#include "Arduino.h"

#include <WiFiNINA.h>
#include "arduino_secrets.h" 
#include "website.h"

WalkieTalkie::WalkieTalkie(HardwareSerial &Serial){
  Serial.begin(9600);      // initialize serial communication
  Serial.println("Walkie Talkie constructor");
  pinMode(9, OUTPUT);      // set the LED pin mode

  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  // attempt to connect to WiFi network:
  while (1) {
    Serial.print("Attempting to connect to Network named: ");
    Serial.println(ssid);                   // print the network name (SSID);
    delay(1000);

    // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
    _status = WiFi.begin(ssid, pass);
    if (_status == WL_CONNECTED) break;
    delay(1000);
  }
  
  for(int i = 0; i < sizeof(data) / sizeof(int); i++){
    data[i / 5][i % 5] = -1;
    if(ROBOT_SERVER_DEBUG){
      Serial.print(i);
      Serial.print(" ");
      Serial.print(i / 5);
      Serial.print(" ");
      Serial.print(i % 5);
      Serial.print(" ");
      Serial.println(data[i / 5][i % 5]);
    }
  }
  _server.begin();
  _printWifiStatus();
  // _waitForClient();
}

void WalkieTalkie::_printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
  // print where to go in a browser:
  Serial.print("To see this page in action, open a browser to http://");
  Serial.println(ip);
}

bool WalkieTalkie::_skipToTrigger(char* t){
  _debugPrintln("Skipping to trigger");
  char* t0 = t;
  int newlineCount = 0;
  while (*t != '\0'){
    if(!_client.available()){
      _debugPrintln("Could not find trigger! Ran out of packet!");
      return false;
    }
    char c = _getNotReturn();

    // Check for trigger
    if (c == *t) t++;
    else t = t0;

    // Check for two newlines
    if (c == '\n') newlineCount++;
    else newlineCount = 0;

    if(newlineCount == 2){
      _debugPrintln("Could not find trigger! Found end!");
      return false;
    }
  }
  _debugPrintln("Found trigger!");
  return true;
}

char WalkieTalkie::_getNotReturn(){
  char ret;
  do {
    ret = _client.read();
  }
  while (ret =='\r');
  if (ROBOT_SERVER_DEBUG) Serial.print(ret);
  return ret;
}

// Return true if commands were received
bool WalkieTalkie::_parseClientData(){
  // Keep going until trigger word read, or return false if end of packet reached
  if (!_skipToTrigger((char*) trigger)) return false;

  // Reset servo pos, duration and colour threshold to -1
  // In Arduino code, just check if the duration isn't -1 
  // to see if a valid command was received
  for(int i = 0; i < sizeof(data) / sizeof(int) / 5; i++){
    data[i][2] = -1;
    data[i][3] = -1;
    data[i][4] = -1;
  }
  
  // This part is not very robust - a badly formed GET request could break things
  // Extra error checks were removed to improve processing speed.
  // Parse strings like '123/4567/8/9/\n' into separate numbers.
  for(int i = 0; i < sizeof(data) / sizeof(int); i++){
    char c = _getNotReturn();
    if (c == '\n' || c == ' ') return i > 0;  // return true if a variable has been modified.
    if (c == '/') continue;       // skip this variable if no numbers sent
    for(int j = 0; j < sizeof(_buf) / sizeof(char); j++){
      if (c == '/'){              // if end of a variable
        _debugPrintln("End of var");
        _buf[j] = '\0';
        break;
      }
      _buf[j] = c;
      c = _getNotReturn();
    }
    // Parse buffer as int
    data[i / 5][i % 5] = atoi(_buf);
  }
  _debugPrintln("Found sufficient variables");
  return true;
}

void WalkieTalkie::_debugPrint(char* x){
  if (ROBOT_SERVER_DEBUG){
    Serial.print(x);
  }
}

void WalkieTalkie::_debugPrintln(char* x){
  if (ROBOT_SERVER_DEBUG){
    Serial.println(x);
  }
}

// Send the website to the _client
void WalkieTalkie::_sendWebsite(int sensorReading){
  _debugPrintln("Sending website...");
  
  // HTTP headers always start with a response code (e.g. HTTP/1.1 200 OK)
  // and a content-type so the _client knows what's coming, then a blank line:
  _client.println("HTTP/1.1 200 OK");
  _client.println("Content-type:text/html");
  _client.println();
  _client.print("TRIGGER/");
  _client.print(sensorReading);
  _client.print("/<br>");
  _client.print(WEBSITE);
  // The HTTP response ends with another blank line:
  _client.println();
}

// Check for new data, parse any new data, and send a website back to the client.
bool WalkieTalkie::listen(int sensorReading){
  _client = _server.available();   // listen for incoming clients
  if (!_client){
    if (ROBOT_SERVER_DEBUG){
      Serial.println("No client.");
      delay(500);
    }
    return false;
  }
  
  _debugPrintln("Client.");
  
  bool ret;
  while(_client.connected()){
    if (_client.available()){
      _debugPrintln("Packet received.");
      
      ret = _parseClientData();
      _debugPrintln("Parsed.");
      
      if (ret){
        if (ROBOT_SERVER_DEBUG){
          Serial.println("Got new data!");
          for(int i = 0; i < sizeof(data) / sizeof(int); i++){
            Serial.println(data[i / 5][i % 5]);
          }
        }
        _skipToTrigger((char*) trigger);
      }
      break;
    }
  }
  _sendWebsite(sensorReading);
  _client.stop();
  return ret;
}