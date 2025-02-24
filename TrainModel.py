import cv2
import numpy as np
import os

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = r"C:\Users\User\Desktop\dataset2"  #Replace with your dataset path
faces = []
labels = []

for user_id, user_folder in enumerate(os.listdir(dataset_path)):
    user_path = os.path.join(dataset_path, user_folder)
    for image_name in os.listdir(user_path):
        img_path = os.path.join(user_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(user_id)

faces = np.array(faces, dtype="object")
labels = np.array(labels)

face_recognizer.train(faces, labels)
face_recognizer.save("trainer.yml")

print("Model trained successfully!")
