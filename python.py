import cv2
import torch
import datetime
import requests
import speech_recognition as sr
import pyttsx3
import time
import os
from ultralytics import YOLO

# ---------------- SETTINGS ---------------- #

CITY = "Ongole"
OPENWEATHER_API = "YOUR_OPENWEATHER_KEY"  # <-- put your key here

KNOWN_WIDTH = 50      # cm
FOCAL_LENGTH = 500    # adjust if needed

# ---------------- INIT ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")

engine = pyttsx3.init()
engine.setProperty("rate", 160)

recognizer = sr.Recognizer()

last_spoken_time = 0
speech_cooldown = 5  # seconds


# ---------------- FUNCTIONS ---------------- #

def speak(text):
    global last_spoken_time
    if time.time() - last_spoken_time > speech_cooldown:
        print("Assistant:", text)
        engine.say(text)
        engine.runAndWait()
        last_spoken_time = time.time()


def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except:
        return ""


def estimate_distance(known_width, focal_length, pixel_width):
    if pixel_width > 0:
        return (known_width * focal_length) / pixel_width
    return -1


def get_time():
    return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}"


def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Temperature is {temp} degree Celsius with {desc}"
        else:
            return "Unable to fetch weather"
    except:
        return "Weather service unavailable"


# ---------------- VOICE COMMAND MODE ---------------- #

speak("Hello! Say start detection to begin.")

while True:
    command = listen()

    if "start detection" in command:
        speak("Detection started.")
        break

    elif "time" in command:
        speak(get_time())

    elif "weather" in command:
        speak(get_weather())

    elif "exit" in command:
        speak("Goodbye")
        exit()


# ---------------- CAMERA DETECTION ---------------- #

cap = cv2.VideoCapture(0)

speak("Camera started.")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_width = x2 - x1

            distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width) / 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, f"{label} {distance:.2f}m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)

            speak(f"{label} detected at {distance:.2f} meters")

    cv2.imshow("Blind Watcher", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()