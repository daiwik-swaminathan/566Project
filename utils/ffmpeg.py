import platform
import subprocess
import threading
import sys
import os
import shutil
import numpy as np
import soundfile as sf

class SystemAudioCapture:
    def __init__(self, samplerate=44100, channels=2, format='s16le', buffer_seconds=10):
        self.samplerate = samplerate
        self.channels = channels
        self.format = format
        self.process = None
        self.running = False
        self.thread = None
        # Add your ffmpeg path to bin here
        self.ffmpeg_path = r"C:\Users\joelp\OneDrive - Cal Poly\CalPoly\CSC 566\ffmpeg\ffmpeg\bin"
        self.bytes_per_sample = 2  # s16le = 2 bytes per sample
        self.buffer_samples = samplerate * buffer_seconds * channels
        self.buffer = np.zeros(self.buffer_samples, dtype=np.int16)
        self.lock = threading.Lock()

    # You can add ffmpeg to the PATH 
    def _patch_path(self):
        if self.ffmpeg_path:
            os.environ['PATH'] = self.ffmpeg_path + os.pathsep + os.environ['PATH']
            print(f"✅ Added {self.ffmpeg_path} to PATH")

    def _get_ffmpeg_command(self):
        system = platform.system()
        if system == 'Windows':
            device = 'audio=Stereo Mix (Realtek(R) Audio)'
            cmd = [
                self.ffmpeg_path + r'\ffmpeg.exe',
                '-f', 'dshow',
                '-i', device,
                '-ac', str(self.channels),
                '-ar', str(self.samplerate),
                '-f', self.format,
                '-'
            ]
        elif system == 'Darwin':
            device = ':0'
            cmd = [
                self.ffmpeg_path + r'\ffmpeg.exe',
                '-f', 'avfoundation',
                '-i', device,
                '-ac', str(self.channels),
                '-ar', str(self.samplerate),
                '-f', self.format,
                '-'
            ]
        elif system == 'Linux':
            cmd = [
                self.ffmpeg_path + r'\ffmpeg.exe',
                '-f', 'pulse',
                '-i', 'default',
                '-ac', str(self.channels),
                '-ar', str(self.samplerate),
                '-f', self.format,
                '-'
            ]
        else:
            raise RuntimeError(f"Unsupported OS: {system}")
        return cmd

    def start(self, callback=None):
        # self._patch_path()
        self.running = True
        print(shutil.which('ffmpeg'))
        cmd = self._get_ffmpeg_command()
        print(f"Running command: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        self.thread = threading.Thread(target=self._read_stream, args=(callback,))
        self.thread.start()

    def _read_stream(self, callback):
        chunk_size = 1024 * self.channels * self.bytes_per_sample
        while self.running:
            data = self.process.stdout.read(chunk_size)
            if not data:
                break
            audio_array = np.frombuffer(data, np.int16)
            with self.lock:
                self.buffer = np.roll(self.buffer, -len(audio_array))
                self.buffer[-len(audio_array):] = audio_array
            if callback:
                callback(data)

    def get_buffer(self):
        with self.lock:
            return np.copy(self.buffer)

    def save_clip(self, filename):
        with self.lock:
            buffer_copy = np.copy(self.buffer)

        if len(buffer_copy) % self.channels != 0:
            trimmed_size = len(buffer_copy) - (len(buffer_copy) % self.channels)
            buffer_copy = buffer_copy[:trimmed_size]

        reshaped = buffer_copy.reshape(-1, self.channels)

        sf.write(filename, reshaped, self.samplerate, format='WAV', subtype='PCM_16')
        print(f"✅ Saved clip to {filename}")

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
        if self.thread:
            self.thread.join()
