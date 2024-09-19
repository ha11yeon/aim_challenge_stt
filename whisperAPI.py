import whisper
import os
import torch

# GPU 설정
# Load Whisper model and move to GPU if available
# Load Whisper model (You can specify "tiny", "base", "small", "medium", or "large")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper 모델을 명시적으로 GPU로 이동
model = whisper.load_model("medium").to(device)

def transcribe_speech(audio_file):
    # Transcribe the audio file using Whisper, set the device to GPU and use FP16 if on GPU
    result = model.transcribe(audio_file, language="ko", fp16=(device == "cuda"))
    print(f"Transcript for {audio_file}: {result['text']}")

def transcribe_all_files_in_directory(directory):
    # Process all .wav files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):  # Only process .wav files
            audio_file_path = os.path.join(directory, file_name)
            print(f"Processing file: {audio_file_path}")
            transcribe_speech(audio_file_path)

if __name__ == "__main__":
    # GPU에서 실행 여부 확인
    print(f"Running on: {device}")
    
    audio_directory = "/raid/co_show02/hayeon/test_audio"  # Path to your audio files
    transcribe_all_files_in_directory(audio_directory)
