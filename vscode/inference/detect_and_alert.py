import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from twilio.rest import Client
import logging

# =============================
# PATHS (CHANGE IF NEEDED)
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FOLDER = os.path.join(BASE_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "VideoActivityModel.pth")

# =============================
# CATEGORIES
# =============================
categories = [
    "assault", "abuse", "harassment", "vandalism",
    "smoking", "burglar", "fight", "arson", "normal"
]

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# RESNET FEATURE EXTRACTOR
# =============================
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_video_features(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    features = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device) # type: ignore

        with torch.no_grad():
            feat = resnet(img_tensor).squeeze().cpu().numpy()

        features.append(feat)
        frame_count += 1

    cap.release()

    if len(features) == 0:
        return np.zeros((1, 512))

    return np.array(features)

# =============================
# LSTM MODEL
# =============================
class VideoClassifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_classes=9):
        super(VideoClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

model = VideoClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# =============================
# TWILIO CONFIG
# =============================
ACCOUNT_SID = "ACf300b2f51ecf75f5bed948363163312d"
AUTH_TOKEN = "2cd25bda9c1a383fcebe8d47e2619c45"

EMERGENCY_CONTACTS = [
    "+916369218864",
    "+916379080914"
]

logging.basicConfig(
    filename="twilio_alerts.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def generate_alert_message(event):
    alerts = {
        "assault": "🚨 Assault detected! Immediate action required.",
        "harassment": "⚠️ Harassment detected. Please verify.",
        "vandalism": "🚔 Vandalism detected."
    }
    return alerts.get(event, "🚨 Suspicious activity detected.")

def send_whatsapp_alert(number, event):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    msg = client.messages.create(
        body=f"ALERT: {generate_alert_message(event)}",
        from_="whatsapp:+14155238886",
        to=f"whatsapp:{number}"
    )
    print(f"✅ WhatsApp alert sent to {number}")
    logging.info(f"Alert sent to {number} | SID: {msg.sid}")

# =============================
# RUN DETECTION
# =============================
alert_sent = False

for file in os.listdir(TEST_FOLDER):
    if not file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    video_path = os.path.join(TEST_FOLDER, file)
    print(f"\nProcessing: {file}")

    features = extract_video_features(video_path)
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    predicted_label = categories[torch.argmax(output).item()] # type: ignore
    print(f"📢 Detected Activity: {predicted_label}")

    if not alert_sent and predicted_label in [
        "assault", "abuse", "harassment", "fight", "burglar"
    ]:
        for contact in EMERGENCY_CONTACTS:
            send_whatsapp_alert(contact, predicted_label)
        alert_sent = True

print("\n✅ Detection completed successfully")
