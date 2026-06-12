import cv2
import pickle
import csv
import os
from datetime import datetime

import os

def mark_attendance(name):
    file_exists = os.path.isfile("attendance.csv")

    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()

        writer.writerow([
            name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

with open("face_data.pkl", "rb") as file:
    data = pickle.load(file)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked = set()
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=8,
        minSize=(100, 100)
    )

    name_found = "Unknown"

    for (x, y, w, h) in faces:
        current_face = gray[y:y+h, x:x+w]
        current_face = cv2.resize(current_face, (100, 100))

        for item in data:
            stored_face = cv2.resize(item["image"], (100, 100))
            diff = cv2.absdiff(stored_face, current_face)
            score = diff.mean()

            if score < 100:
                name_found = item["name"]

                if name_found not in marked:
                    mark_attendance(name_found)
                    marked.add(name_found)

                break

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name_found, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()