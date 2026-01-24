from inference.predict import predict_video
import os

video_path = "test_videos/sample1.mp4"

result = predict_video(video_path)
print("Prediction:", result)
