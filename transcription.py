# transcription.py

from youtube_transcript_api import YouTubeTranscriptApi # type: ignore

def get_transcript(video_url):
    try:
        # Extract video ID from URL
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1]
        else:
            raise ValueError("Invalid YouTube URL")

        # Get transcript list (each part is a timestamped chunk)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Join all chunks into a single string
        transcript = " ".join([entry["text"] for entry in transcript_list])
        return transcript

    except Exception as e:
        raise RuntimeError(f"Failed to get transcript: {e}")
