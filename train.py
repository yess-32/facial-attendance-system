import cv2
import os
import pickle

faces_dir = "faces"
data = []

for student_name in os.listdir(faces_dir):
    student_folder = os.path.join(faces_dir, student_name)

    if not os.path.isdir(student_folder):
        continue

    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append({
            "name": student_name,
            "image": gray
        })

with open("face_data.pkl", "wb") as file:
    pickle.dump(data, file)

print("Training completed!")
print("Total images trained:", len(data))