#include <Arduino.h>
#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#include <BME280I2C.h>
#include <Wire.h>

#include "DHT.h"

#include <NTPClient.h>
#include <Time.h>
#include <TimeLib.h>
#include <Timezone.h>

#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

#include "esp_system.h"

#define WIFI_SSID "TSM-FIBRAtBkrBc_EXT"
#define WIFI_PASSWORD "zPex38zU"

#define API_KEY "AIzaSyD8kPC8bpxndwdIslt_1oNfPiI_k3sIDik"

#define DHTPIN 4
#define DHTTYPE DHT22

#define NTP_OFFSET 60 * 60
#define NTP_INTERVAL 60 * 1000
#define NTP_ADDRESS "pool.ntp.org"

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, NTP_ADDRESS, NTP_OFFSET, NTP_INTERVAL);
TimeChangeRule CEST = { "CEST", Last, Sun, Oct, 3, 60 };
Timezone esTZ(CEST, CEST);
time_t local, utc;

FirebaseAuth auth;
FirebaseConfig config;

FirebaseData firebaseData;

BME280I2C bme;

DHT dht(DHTPIN, DHTTYPE);

float temp, hum, pres;

bool signupOK = false;


const char* months[] = { "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic" };

int anemometro_pin = 15;
volatile int revoluciones = 0;
unsigned long oldmillis = 0;
int UPLOADTIME = 60000;
bool lastState = LOW;
unsigned long lastTime = 0;

int pin_intesidadluminica = 33;

// Variables y constantes para el modelo físico
const float Cd = 0.47;             // Coeficiente de arrastre para una esfera, ajustar según la forma de las aspas
const float A = 0.0033183;                  // Área proyectada de las aspas en m^2, ajustar según tus aspas
float rho;                
const float ajusteEmpirico = 1.1;  // Factor de ajuste empírico
const float masaAspas = 0.065;
const float errorEmpirico = 0.90;

const int led_azul=19; 
const int led_rojo = 23;

unsigned long previousMillis = 0;
const unsigned long interval = 1800000;

void setup() {
  Serial.begin(115200);
  btStop();

  //Tierra de Prueba
  pinMode(2, OUTPUT);
  digitalWrite(2, LOW);
  pinMode(18, OUTPUT);
  digitalWrite(18, LOW);
  pinMode(5, OUTPUT);
  digitalWrite(5, LOW);

  Serial.print("READY");

  pinMode(led_azul, OUTPUT);
  pinMode(led_rojo, OUTPUT);

  pinMode(pin_intesidadluminica, INPUT);

  pinMode(anemometro_pin, INPUT);

  Wire.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("WIFI CONECTADA");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println();


  config.api_key = API_KEY;

  config.database_url = "https://invg-proyecto-default-rtdb.firebaseio.com/";

  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("ok");
    signupOK = true;
  } else {
    Serial.printf("%s\n", config.signer.signupError.message.c_str());
  }

  /* Assign the callback function for the long running token generation task */
  config.token_status_callback = tokenStatusCallback;  //see addons/TokenHelper.h

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  while (!bme.begin()) {
    Serial.println("No se pudo encontrar un BME conectado");
    delay(1000);
  }
  Serial.println("BME280 Encontrado!");

  dht.begin();

  delay(3000);
}

String Timeahora() {
  timeClient.update();  // Actualizar el cliente NTP y obtener la marca de tiempo UNIX UTC
  unsigned long utc = timeClient.getEpochTime();
  local = esTZ.toLocal(utc);

  String date = "";
  date += day(local);
  date += "-";
  date += months[month(local) - 1];
  date += "-";
  date += year(local);

  String hora = "";  // Funcion para formatear en texto la hora
  if (hour(local) < 10)
    hora += "0";
  hora += hour(local);
  hora += ":";
  if (minute(local) < 10)  // Agregar un cero si el minuto es menor de 10
    hora += "0";
  hora += minute(local);

  return date + "_" + hora;
}

String weather_txt() {
  HTTPClient http;

  http.begin("http://api.openweathermap.org/data/2.5/weather?q=Puerto%20de%20Mazarr%C3%B3n,es&appid=1a6f24977c77e75edaad7f4ee40a7876&lang=es");
  http.addHeader("User-Agent", "ESP32 HTTP Client");

  int httpResponseCode = http.GET();

  if (httpResponseCode > 0 && httpResponseCode == 200) {
    String payload = http .getString();
    //Serial.println(payload);  // Depuración del payload recibido

    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload);

    if (!error) {
      // Acceder a la descripción del clima
      const char* description = doc["weather"][0]["description"];
      //Serial.println(description);
      return String(description);
    } else {
      Serial.print("Error al parsear JSON: ");
      Serial.println(error.c_str());
    }

  } else {
    Serial.print("Error en la solicitud HTTP: ");
    Serial.println(httpResponseCode);
  }

  http.end();
}

void checkFirebaseAndReconnect() {
  if (!Firebase.ready()) {
    Serial.println("Firebase desconectado, reconectando...");
    Firebase.begin(&config, &auth);
  }
}

void checkWiFiAndReconnect() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi desconectado, reconectando...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
      delay(1000);
      Serial.print(".");
    }
    Serial.println("WiFi reconectado.");
  }
}

String prediccion(float temperatura, float humedad, float presion, float velocidad, float luz){
  HTTPClient http;

  String url = "https://invgproy.onrender.com/predict?hum=" + String(humedad) + "&luz=" + String(luz) + "&pres=" + String(presion) + "&temp=" + String(temperatura) + "&vel=" + velocidad;
  

  http.begin(url);
  http.addHeader("User-Agent", "ESP32 HTTP Client");

  http.setTimeout(600000);

  int httpResponseCode = http.GET();

  if (httpResponseCode > 0 && httpResponseCode == 200) {
    String payload = http.getString();
    //Serial.println(payload);  // Depuración del payload recibido

    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload);

    if (!error) {
      // Acceder a la predicción del clima
      const char* prediction = doc["prediction"];
      //Serial.println(prediction);
      return String(prediction);
    } else {
      Serial.print("Error al parsear JSON: ");
      Serial.println(error.c_str());
      return "Error al parsear JSON";
    }

  } else {
    Serial.print("Error en la solicitud HTTP: ");
    Serial.println(httpResponseCode);
    return "Error en la solicitud HTTP";
  }

  http.end();
}

void firebaseRW(float temperatura, float humedad, float presion, float velocidad) {
  String TimeNow = Timeahora();
  Serial.println(TimeNow);

  String ruta_con_fecha_hum = "/" + TimeNow + "/HUM";
  String ruta_con_fecha_temp = "/" + TimeNow + "/TEMP";
  String ruta_con_fecha_pres = "/" + TimeNow + "/PRES";
  String ruta_con_fecha_vel = "/" + TimeNow + "/VEL";
  String ruta_con_fecha_weather = "/" + TimeNow + "/WEATHER";
  String ruta_con_fecha_luz = "/" + TimeNow + "/LUZ";
  String ruta_con_fecha_pred = "/" + TimeNow + "/PRED";

  float luz = (map(analogRead(pin_intesidadluminica), 0, 4095, 0, 10.0));
  String weather = weather_txt();
  String predic = prediccion(humedad, temperatura, presion, velocidad, luz);

  Firebase.RTDB.setInt(&firebaseData, ruta_con_fecha_hum, humedad);
  Firebase.RTDB.setInt(&firebaseData, ruta_con_fecha_temp, temperatura);
  Firebase.RTDB.setInt(&firebaseData, ruta_con_fecha_pres, presion);
  Firebase.RTDB.setFloat(&firebaseData, ruta_con_fecha_vel, velocidad);
  Firebase.RTDB.setFloat(&firebaseData, ruta_con_fecha_luz, luz);
  Firebase.RTDB.setString(&firebaseData, ruta_con_fecha_weather, weather);
  Firebase.RTDB.setString(&firebaseData, ruta_con_fecha_pred, predic);

  Firebase.RTDB.setInt(&firebaseData, "/LIVE/HUM", humedad);
  Firebase.RTDB.setInt(&firebaseData, "/LIVE/TEMP", temperatura);
  Firebase.RTDB.setInt(&firebaseData, "/LIVE/PRES", presion);
  Firebase.RTDB.setInt(&firebaseData, "/LIVE/VEL", velocidad);
  Firebase.RTDB.setFloat(&firebaseData, "/LIVE/LUZ", luz);
  Firebase.RTDB.setString(&firebaseData, "/LIVE/WEATHER", weather);
  Firebase.RTDB.setString(&firebaseData, "/LIVE/PRED", predic);

  Serial.print("Datos SUBIDOS");
  digitalWrite(led_azul, HIGH);
  delay(1500);
  digitalWrite(led_azul, LOW);
}

void IRAM_ATTR cuentaRevoluciones() {
  bool currentState = digitalRead(anemometro_pin);
  if (currentState != lastState) {
    if (currentState == HIGH) {
      revoluciones++;
    }
  }
  lastState = currentState;
}

void loop() {
  if (millis() - previousMillis >= interval) {
    previousMillis = millis();
    esp_restart();
  }

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(led_rojo, HIGH); 
  } else {
    digitalWrite(led_rojo, LOW); 
  }

  checkWiFiAndReconnect();
  checkFirebaseAndReconnect();

  attachInterrupt(digitalPinToInterrupt(anemometro_pin), cuentaRevoluciones, CHANGE);

  if ((millis() - oldmillis) >= UPLOADTIME) {
    //Serial.println(millis() - oldmillis);
    float longitud = 0.095 * (revoluciones*errorEmpirico);

    float vel = longitud / ((millis() - oldmillis) / 1000);

    Serial.println(revoluciones);
    revoluciones = 0;
    oldmillis = millis();

    float h = dht.readHumidity();

    float t = dht.readTemperature();

    bme.read(pres, temp, hum);

    rho = (pres*100)/(287.05*(t+273));

    float F_res = 0.5 * Cd * A * rho * pow(vel, 2);
    float velCorregida = vel - (F_res / (masaAspas * ajusteEmpirico));

    velCorregida = max(velCorregida, 0.0f);

    Serial.print("Esto es velcorregida:");
    Serial.println(velCorregida);

    firebaseRW(t, h, pres, velCorregida);
  }

  delay(400);
}
