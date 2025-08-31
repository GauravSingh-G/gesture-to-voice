from gtts import gTTS
import pygame
import tempfile
import time
import os

# Generate speech
tts = gTTS("APPLE", lang='en')
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
tmp_path = tmp.name
tmp.close()  # Close before writing
tts.save(tmp_path)

# Play using pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(tmp_path)
pygame.mixer.music.play()

print(f"🔊 Playing: {tmp_path}")
while pygame.mixer.music.get_busy():
    time.sleep(0.5)

# Clean up properly
pygame.mixer.music.stop()
pygame.mixer.quit()
pygame.quit()
os.remove(tmp_path)
