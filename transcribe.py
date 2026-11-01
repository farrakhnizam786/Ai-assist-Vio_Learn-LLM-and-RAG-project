import json
import os

import whisper
from moviepy import VideoFileClip


def convert_video_to_mp3(video_path, audio_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    os.makedirs(os.path.dirname(os.path.abspath(audio_path)), exist_ok=True)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    print(f"[INFO] Extracting audio from '{video_path}'...")
    video_clip = VideoFileClip(video_path)
    try:
        if video_clip.audio is None:
            raise ValueError("The selected video does not contain an audio track.")
        video_clip.audio.write_audiofile(audio_path, codec="libmp3lame")
    finally:
        video_clip.close()

    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("Audio extraction did not produce a usable audio file.")
    return audio_path


def transcribe_and_optimize_chunks(audio_path, json_path):
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        raise FileNotFoundError("The extracted audio file is missing or empty.")

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    if os.path.exists(json_path):
        os.remove(json_path)

    print("[INFO] Loading Whisper base model...")
    model = whisper.load_model("base")
    print(f"[INFO] Transcribing audio: {audio_path}...")
    result = model.transcribe(audio_path)

    optimized_chunks = []
    current_chunk_text = ""
    raw_segments = result.get("segments") or []
    start_time = raw_segments[0]["start"] if raw_segments else 0.0
    end_time = start_time

    for segment in raw_segments:
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue
        current_chunk_text = f"{current_chunk_text} {segment_text}".strip()
        end_time = segment["end"]

        if len(current_chunk_text) >= 400:
            optimized_chunks.append(
                {
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": current_chunk_text,
                }
            )
            current_chunk_text = ""
            start_time = end_time

    if current_chunk_text:
        optimized_chunks.append(
            {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": current_chunk_text,
            }
        )

    if not optimized_chunks:
        raise ValueError(
            "No speech could be transcribed from this video. "
            "Please provide another video with clear spoken content."
        )

    with open(json_path, "w", encoding="utf-8") as transcript_file:
        json.dump(optimized_chunks, transcript_file, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Transcript chunks saved to '{json_path}'")
    return optimized_chunks
