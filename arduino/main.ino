#include "robot_server.h"
#include <Adafruit_MotorShield.h>
#include <arduino-timer.h>
#include <Servo.h>

// Timers
auto timer = timer_create_default();
auto amberTimer = timer_create_default();
auto greenRedLEDTimer = timer_create_default();

// Objects
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *motorleft = AFMS.getMotor(1);
Adafruit_DCMotor *motorright = AFMS.getMotor(2);
Servo gate;

// State variables
int ml = 0;
int mr = 0;
int pos = 45;
int dur = -1;
int colourThresh = -1;

int index = 0;

// Sensors / lights
int sensorPin = A0;
int amberLEDPin = 2;
int redLEDPin = 1;
int greenLEDPin = 0;
int amberLEDState = 0x1; // 1 = on

int lastReading = 0;
float avgScale = 0.95;

void initMotors(){
  motorleft->run(RELEASE);
  motorright->run(RELEASE);
  gate.attach(9);
  gate.write(pos);
}

void setMotor(Adafruit_DCMotor* motor, int speed){
  motor->run(RELEASE);
  if (speed == 0){
    return;
  }
  if (speed < 0){
    motor->setSpeed(-speed); 
    motor->run(BACKWARD);
  }
  else {
    motor->setSpeed(speed);
    motor->run(FORWARD);
  }
}

// Execute the commands as per the current values of ml, mr, etc.
void updateState(){
  // Spin motors
  setMotor(motorleft, ml);
  setMotor(motorright, mr);
  // Spin servo
  if(pos != -1) gate.write(pos);
  // Show LED colour
  if(colourThresh != -1){
    if(lastReading > colourThresh){
      digitalWrite(redLEDPin, HIGH);
    }
    
    else{
      digitalWrite(greenLEDPin, HIGH);
    }
    greenRedLEDTimer.in(5000, turnOffGreenRedLEDs);
  }
}

// Get the variables at current index
// Returns false if the duration is invalid
bool updateVariables(WalkieTalkie &wt){
  // If duration is -1, do nothing
  if(wt.data[index][3] == -1) return false;
  // Else update vars
  ml = -wt.data[index][0];
  mr = -wt.data[index][1];
  // Don't reset the servo position
  pos = wt.data[index][2];
  dur = wt.data[index][3];
  colourThresh = wt.data[index][4];
  return true;
}

void stop(){
  index = 0;
  ml = 0;
  mr = 0;
  colourThresh = -1;
  updateState();
}

void doNext(WalkieTalkie *wt){
  // Don't read outside the array
  if(index >= 1){
    stop();
    return;
  }
  // Stop if the command is invalid
  index += 1;
  if(!updateVariables(*wt)) {
    stop();
    return;
  }
  // Update state, and set the timer again
  updateState();
  timer.in(dur, doNext, wt);
}

void flipAmberLEDState(void*){
  digitalWrite(amberLEDPin, amberLEDState);
  amberLEDState = 1 - amberLEDState;
  Serial.println(amberLEDState);
}

void turnOffGreenRedLEDs(void*){
  digitalWrite(greenLEDPin, LOW);
  digitalWrite(redLEDPin, LOW);
}

void setup() {
  // Init LEDs
  pinMode(amberLEDPin, OUTPUT);
  pinMode(redLEDPin, OUTPUT);
  pinMode(greenLEDPin, OUTPUT);
  WalkieTalkie wt(Serial);

  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");
  initMotors();
  amberTimer.every(250, flipAmberLEDState);
  while(1){
    timer.tick();
    amberTimer.tick();
    greenRedLEDTimer.tick();
    int reading = analogRead(sensorPin);
    lastReading = lastReading * avgScale + reading * (1 - avgScale);
    if(wt.listen(lastReading)){
      if(!updateVariables(wt)) continue;
      updateState();
      if (dur > 0){
        // Clear any existing timers, and set a new one
        timer.cancel();
        timer.in(dur, doNext, &wt); // Passing pointers to objects makes me sad
      }
      Serial.println("New data found");
      for(int i = 0; i < sizeof(wt.data) / sizeof(int); i++){
        Serial.println(wt.data[i / 5][i % 5]);
      }
    }
  }
}

void loop() {
}
