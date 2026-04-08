import os
import subprocess
from moviepy import ColorClip, AudioFileClip

# 1. Generate local audio files using macOS "say" command
clean_text = "Hello everyone! This is a completely normal and nice statement. I hope you have a wonderful day."
toxic_text = "What a complete idiot! I hate this so much. You are all a bunch of losers."

print("Generating synthetic speech audio via macOS 'say'...")
subprocess.run(['say', clean_text, '-o', 'clean_part.aiff'])
subprocess.run(['say', toxic_text, '-o', 'toxic_part.aiff'])

# We can merge them via moviepy
print("Creating sample MP4 video using MoviePy...")
from moviepy import concatenate_audioclips

audio1 = AudioFileClip("clean_part.aiff")
audio2 = AudioFileClip("toxic_part.aiff")

# Combine the two clips
final_audio = concatenate_audioclips([audio1, audio2])

# Create a blank video with a generic color, same duration as the audio
video = ColorClip(size=(640, 480), color=(50, 50, 150), duration=final_audio.duration)

# Set the audio of the video
video = video.with_audio(final_audio)

# Write out the file
output_path = "sample_test_video.mp4"
video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")

# Cleanup intermediate aiff files
os.remove("clean_part.aiff")
os.remove("toxic_part.aiff")

print(f"\nSample video successfully created at: {os.path.abspath(output_path)}")
