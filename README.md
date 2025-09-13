# 🔫 Weapon Detection System Using YOLOv8

This project uses **YOLOv3** and **OpenCV** to detect weapons (gun, knife, pistol, etc.) in real-time via webcam. When a weapon is detected, the frame is saved and an alert is triggered.

---

## 📁 Project Structure

weapon_detection/
├── yolov3_training_2000.weights # Trained weights file (download link below)
├── yolov3_testing.cfg # YOLOv3 config file
├── detections/ # Saved alert frames
├── weapon_detection.py # Main detection script
└── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-repo/weapon-detection.git
cd weapon-detection
Install dependencies

bash
Copy
Edit
pip install opencv-python numpy playsound
Download YOLOv3 Trained Files

Download yolov3_training_2000.weights and yolov3_testing.cfg from this link:

📥 Download Trained Model

Place them in the same folder as your script or update the path in weapon_detection.py.

Run the Script

bash
Copy
Edit
python weapon_detection.py
🧠 Features
Real-time detection from webcam

Boxes in 🔴 red for weapons, 🟢 green for other objects

FPS display for performance

Saves weapon detection frames in /detections

🔐 Optional: Email Alert (Config Required)
This feature is enabled but optional. Add your credentials in the script if you want to receive alerts via email.

python
Copy
Edit
EMAIL_SENDER = "youremail@example.com"
EMAIL_PASSWORD = "yourpassword"
EMAIL_RECEIVER = "receiver@example.com"
📸 Example

🛠 Notes
You can replace cap = cv2.VideoCapture(1) with 0 if your default webcam index is different.

Improve performance by running on GPU (if available) via OpenCV DNN CUDA backend.

✅ To Do
 Sound alerts

 GUI display

 Automatic video recording of detection

📄 License
MIT License - Free to use and modify.

yaml
Copy
Edit

---

