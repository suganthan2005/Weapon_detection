import cv2
import numpy as np
import os
import datetime
import smtplib
import time
import playsound


# Load YOLOv3 weights and configuration

net = cv2.dnn.readNet(
    #USE YOUR PATH HERE
   # "./weapon/weapon_detection/yolov3_training_2000.weights",  download from https://drive.usercontent.google.com/download?id=10uJEsUpQI3EmD98iwrwzbD4e19Ps-LHZ&export=download&authuser=0
    #"./weapon/weapon_detection/yolov3_testing.cfg"
)

# Define classes and known weapon labels

classes = ["weapon", "knife", "gun", "rifle", "pistol"]
weapon_classes = set(["weapon", "knife", "gun", "rifle", "pistol"])
colors = np.random.uniform(0, 255, size=(len(classes) + 10, 3))  # allow colors for more objects


# Get output layer names

layer_names = net.getUnconnectedOutLayersNames()

# ---------------------------------------
# Email settings
# ---------------------------------------
EMAIL_SENDER = "youremail@example.com"
EMAIL_PASSWORD = "yourpassword"
EMAIL_RECEIVER = "receiver@example.com"

def send_email_alert(object_name):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            message = f"Subject: Weapon Alert!\n\nA weapon has been detected: {object_name.upper()}"
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message)
            print("[EMAIL] Alert sent.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

def save_detection_frame(frame, label):
    os.makedirs("detections", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detections/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[SAVE] Frame saved: {filename}")

# ---------------------------------------
# Start video capture with lower resolution for speed
# ---------------------------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)            # Cap frame rate for better FPS

if not cap.isOpened():
    print("[ERROR] Webcam not accessible.")
    exit()

print("[INFO] Weapon detection started. Press ESC to stop.")

# ---------------------------------------
# Frame processing loop
# ---------------------------------------
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            label = classes[class_id] if class_id < len(classes) else "unknown"
            color = (0, 0, 255) if label in weapon_classes else (0, 255, 0)

            if label in weapon_classes:
                save_detection_frame(frame, label)
                
                send_email_alert(label)
                print(f"[ALERT] Weapon detected: {label.upper()}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label.upper(), (x, y - 10), font, 0.8, color, 2)

    # Calculate and show FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 0.7, (255, 255, 0), 2)

    cv2.imshow("Weapon Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
