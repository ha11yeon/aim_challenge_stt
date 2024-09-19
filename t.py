import whisper
import os
import torch

# Load Whisper model and move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)

# Check if model is on GPU
print(f"Model is running on: {next(model.parameters()).device}")
print("hello world")
print(torch.cuda.is_available())  # True이면 GPU 사용 가능
print(torch.cuda.current_device())  # GPU 장치 번호 확인
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU 장치 이름 확인
