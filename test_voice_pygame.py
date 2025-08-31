from gtts import gTTS
import pygame
import time
import tempfile
import os

text = "Testing one two three"

# Create temp file path
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
    temp_path = fp.name

# Save the MP3
tts = gTTS(text=text, lang='en')
tts.save(temp_path)

# Init and play using pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(temp_path)
pygame.mixer.music.play()
print("🔊 Playing:", text)

# Wait until done
while pygame.mixer.music.get_busy():
    time.sleep(0.1)

pygame.mixer.music.stop()
pygame.quit()

# Remove the file
os.remove(temp_path)
