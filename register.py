import cv2
import os

name = input("Enter student name: ")

folder = f"faces/{name}"
os.makedirs(folder, exist_ok=True)

camera = cv2.VideoCapture(0)

count = 0

while count < 20:
    ret, frame = camera.read()

    cv2.imshow("Register Face", frame)

    cv2.imwrite(f"{folder}/{count}.jpg", frame)

    count += 1

    if cv2.waitKey(300) == 27:
        break

camera.release()
cv2.destroyAllWindows()

print("Registration Complete!")