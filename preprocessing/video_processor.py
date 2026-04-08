import os
import whisper
import warnings
from moviepy import VideoFileClip

# Suppress some ffmpeg/whisper warnings
warnings.filterwarnings("ignore")

class VideoProcessor:
    def __init__(self, model_size="base"):
        """
        Initializes the VideoProcessor.
        model_size can be 'tiny', 'base', 'small', 'medium', or 'large'.
        'base' is a good tradeoff between speed and accuracy for testing.
        """
        print(f"Loading Whisper model ({model_size})... This might take a moment the first time.")
        self.model = whisper.load_model(model_size)
        
    def extract_audio(self, video_path, output_audio_path):
        """
        Extracts audio from a video file and saves it as a WAV file.
        """
        try:
            print(f"Extracting audio from {video_path}...")
            video = VideoFileClip(video_path)
            
            # Write audio to file, suppressing output
            video.audio.write_audiofile(output_audio_path, logger=None)
            video.close()
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def transcribe_audio(self, audio_path):
        """
        Transcribes the given audio file using Whisper.
        Returns the raw transcribed text and segment-level data.
        """
        try:
            print(f"Transcribing audio from {audio_path}...")
            # For simplicity, we just use transribe without language enforcement,
            # though it usually auto-detects nicely.
            result = self.model.transcribe(audio_path)
            
            # The result dict contains 'text' (full text) and 'segments'
            return {
                "text": result["text"].strip(),
                "segments": result["segments"]
            }
        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"text": "", "segments": []}

    def process_video(self, video_path):
        """
        End-to-end processing: Video -> Audio -> Text
        """
        basename = os.path.basename(video_path)
        name, _ = os.path.splitext(basename)
        
        # We will save temporary audio files in a cache or same dir
        output_dir = os.path.dirname(video_path)
        audio_path = os.path.join(output_dir, f"{name}.wav")
        
        success = self.extract_audio(video_path, audio_path)
        if not success:
            return None
            
        transcription = self.transcribe_audio(audio_path)
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return transcription

if __name__ == "__main__":
    # Simple test stub
    vp = VideoProcessor("tiny")
    print("Video Processor initialized.")
