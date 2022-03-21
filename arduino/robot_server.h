#ifndef ROBOT_SERVER_H
#define ROBOT_SERVER_H
#define ROBOT_SERVER_DEBUG false
#include "Arduino.h"

#include "WiFiNINA.h"
#include "arduino_secrets.h" 
#include "website.h"

const char ssid[] = SECRET_SSID;
const char pass[] = SECRET_PASS;
const char trigger[] = "GET /TRIGGER/";     // marker designating the start of data

class WalkieTalkie {
    public:
        WalkieTalkie(HardwareSerial &Serial);
        // (Motor left, motor right, servo pos, duration, colour threshold) x2
        int data[2][5] = {{0,0,0,500,0}, {0,0,0,500,0}};        // parsed data from client
        bool shouldBlinkLED = false;

        // Check for new data, parse any new data, and send a website back to the client
        bool listen(int sensorReading);
    private:
        // int keyIndex = 0;                // your network key index number (needed only for WEP)

        int _status = WL_IDLE_STATUS;      
        char _buf[255];                    // temporary buffer to store data chunks

        int _port = 80;

        WiFiServer _server{80};
        WiFiClient _client;

        void _printWifiStatus();

        // Get the next character in the packet that isn't a carriage return
        char _getNotReturn();

        // Return true if new commands were received
        bool _parseClientData();

        // Return true if trigger was found in packet
        bool _skipToTrigger(char* t);

        // Send the website to the client
        void _sendWebsite(int sensorReading);

        void _debugPrint(char* x);
        void _debugPrintln(char* x);
};

// Ignore the "PCH warning" error here
#endif