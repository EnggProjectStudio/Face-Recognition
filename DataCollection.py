import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = r"C:\Users\User\Desktop\dataset2"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

user_id = input("Enter a unique user ID: ")
user_folder = os.path.join(dataset_path, user_id)
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

cap = cv2.VideoCapture(0)

count = 0
max_samples = 150

print(f"Collecting face data for user ID: {user_id}")

while count < max_samples:
    ret, img = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        img_path = os.path.join(user_folder, f"{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        count += 1

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"Collected {count} face images for user ID: {user_id}")
