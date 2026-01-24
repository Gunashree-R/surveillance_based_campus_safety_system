import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from twilio.rest import Client
import requests
import threading
from tkinter import Tk, filedialog
import os

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ CATEGORIES ------------------
categories = [
    "assault", "abuse", "harassment", "vandalism",
    "smoking", "burglar", "fight", "arson", "normal"
]

# ------------------ FEATURE EXTRACTOR ------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def extract_video_features(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    features = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device) # type: ignore

        with torch.no_grad():
            feat = resnet(img_tensor).squeeze().cpu().numpy()

        features.append(feat)
        count += 1

    cap.release()
    return np.array(features) if len(features) > 0 else None

# ------------------ LSTM MODEL ------------------
class VideoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, len(categories))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ------------------ LOAD MODEL ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "VideoActivityModel.pth")

model = VideoClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully")

# ------------------ TWILIO ------------------
ACCOUNT_SID = "ACf300b2f51ecf75f5bed948363163312d"
AUTH_TOKEN = "2cd25bda9c1a383fcebe8d47e2619c45"
TWILIO_WHATSAPP = "whatsapp:+14155238886"
EMERGENCY_CONTACTS = [
    "whatsapp:+916369218864",
    "whatsapp:+916379080914"
]

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_whatsapp_alert(activity):
    message = f"🚨 ALERT: {activity.upper()} detected!"
    for contact in EMERGENCY_CONTACTS:
        client.messages.create(
            from_=TWILIO_WHATSAPP,
            to=contact,
            body=message
        )
        print(f"✅ WhatsApp sent to {contact}")

# ------------------ ESP32 ------------------
ESP32_URL = "https://xxxx.ngrok-free.dev/alert"

def trigger_buzzer():
    try:
        requests.get(ESP32_URL, timeout=5)
        print("✅ ESP32 buzzer triggered")
    except Exception as e:
        print("⚠️ ESP32 error:", e)

def trigger_alerts(activity):
    t1 = threading.Thread(target=send_whatsapp_alert, args=(activity,))
    t2 = threading.Thread(target=trigger_buzzer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# ------------------ VIDEO UPLOAD ------------------
Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select a video for activity detection",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
)

if not video_path:
    print("❌ No video selected")
    exit()

print(f"\n📂 Selected video: {video_path}")

# ------------------ PREDICTION ------------------
features = extract_video_features(video_path)

if features is None:
    print("❌ No frames extracted")
    exit()

tensor = torch.tensor(features).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(tensor)
    predicted_label = categories[output.argmax().item()]

print(f"\n📢 Detected Activity: {predicted_label}")

# ------------------ ALERT ------------------
if predicted_label in ["assault", "abuse", "harassment", "fight", "arson"]:
    trigger_alerts(predicted_label)
else:
    print("✅ Normal activity detected")

print("\n✅ Upload check completed successfully")
