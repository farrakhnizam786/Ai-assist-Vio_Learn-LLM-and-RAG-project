import os
import uuid

from rag_pipeline import (
    NON_EDUCATIONAL_MESSAGE,
    classify_educational_content,
    generate_ai_tutor_response,
    retrieve_top_chunks,
    store_chunks_in_chroma,
)
from transcribe import convert_video_to_mp3, transcribe_and_optimize_chunks


if __name__ == "__main__":
    print("==================================================")
    print("       AI TEACHING ASSISTANT (LIVE TERMINAL)      ")
    print("==================================================")

    input_dir = "input_videos"
    output_dir = "output_data"
    db_dir = "vector_db"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    video_name = input(
        "Enter video file name inside 'input_videos/': "
    ).strip()
    video_path = os.path.join(input_dir, video_name)
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: '{video_path}'")
        raise SystemExit(1)

    video_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    try:
        audio_path = os.path.join(output_dir, f"{video_id}.mp3")
        audio_file = convert_video_to_mp3(video_path, audio_path)
        json_path = os.path.join(output_dir, f"{video_id}.json")
        chunks = transcribe_and_optimize_chunks(audio_file, json_path)

        educational, _reason = classify_educational_content(chunks, video_name)
        if not educational:
            print(NON_EDUCATIONAL_MESSAGE)
            raise SystemExit(1)

        collection = store_chunks_in_chroma(
            chunks,
            db_path=db_dir,
            video_id=video_id,
            session_id=session_id,
        )
    except SystemExit:
        raise
    except Exception as error:
        print(f"[ERROR] Could not process this video: {error}")
        raise SystemExit(1) from error

    print("\n--------------------------------------------------")
    print("🎓 AI Teaching Assistant: Hello student!")
    print("📁 Your video was transcribed and indexed.")
    print("--------------------------------------------------")

    user_intent = input(
        "\n❓ Do you have any question related to this video? (yes/no): "
    ).strip().lower()
    if user_intent in {"yes", "y", "ha", "haan"}:
        student_query = input("✍️ Please write here: ").strip()
        if not student_query:
            student_query = "What is the main concept explained in this video?"
    else:
        print("\n[INFO] Preparing an overview of this video...")
        student_query = "What is the main concept explained in this video?"

    try:
        top_chunks = retrieve_top_chunks(
            student_query,
            collection,
            video_id=video_id,
            session_id=session_id,
            top_k=3,
        )
        final_answer = generate_ai_tutor_response(student_query, top_chunks)
    except Exception as error:
        print(f"[ERROR] Could not answer this question: {error}")
        raise SystemExit(1) from error

    print("\n==================================================")
    print("                 AI TUTOR ANSWER                  ")
    print("==================================================")
    print(final_answer)
    print("==================================================")
