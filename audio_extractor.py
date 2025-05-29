from pydub import AudioSegment
import os

# === Input settings ===
input_path = "nba_1.mp3"  # Change to your actual mp3 file
output_dir = "split_output"
chunk_length_ms = 10 * 1000  # 10 seconds in milliseconds

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load audio ===
audio = AudioSegment.from_mp3(input_path)
total_length = len(audio)

# === Split and save chunks ===
for i in range(0, total_length, chunk_length_ms):
    chunk = audio[i:i + chunk_length_ms]
    chunk_name = f"chunk_{i // chunk_length_ms:03d}.mp3"
    chunk_path = os.path.join(output_dir, chunk_name)
    chunk.export(chunk_path, format="mp3")
    print(f"Saved: {chunk_path}")

print("Done splitting.")