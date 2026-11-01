import hashlib
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import yt_dlp

from rag_pipeline import (
    NON_EDUCATIONAL_MESSAGE,
    classify_educational_content,
    generate_ai_tutor_response,
    get_chroma_collection,
    retrieve_top_chunks,
    store_chunks_in_chroma,
)
from transcribe import convert_video_to_mp3, transcribe_and_optimize_chunks

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input_videos"
OUTPUT_DIR = PROJECT_DIR / "output_data"
DB_DIR = PROJECT_DIR / "vector_db"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_INPUT_DIR = INPUT_DIR / st.session_state.session_id
SESSION_INPUT_DIR.mkdir(parents=True, exist_ok=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "current_video_path" not in st.session_state:
    st.session_state.current_video_path = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "video_ids" not in st.session_state:
    st.session_state.video_ids = {}
if "last_selected_video" not in st.session_state:
    st.session_state.last_selected_video = None


def download_video_from_url(url, download_dir):
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Enter a valid HTTP or HTTPS video URL.")

    download_id = uuid.uuid4().hex
    output_template = str(download_dir / f"downloaded_{download_id}.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "overwrites": True,
        "extractor_args": {"youtube": {"player_client": ["default"]}},
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url.strip(), download=True)
            requested_downloads = info.get("requested_downloads") or []
            candidates = [
                Path(download["filepath"])
                for download in requested_downloads
                if download.get("filepath")
            ]
            candidates.append(Path(downloader.prepare_filename(info)))
            candidates.extend(download_dir.glob(f"downloaded_{download_id}.*"))
    except Exception as error:
        raise RuntimeError(
            "Download failed. Confirm the URL is valid and the video platform is supported."
        ) from error

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("The download completed without producing a video file.")


with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Control Panel")
    st.markdown("Upload a video or paste a video link to start.")
    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload a video",
        type=["mp4", "mov", "avi", "webm", "mkv"],
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        extension = Path(uploaded_file.name).suffix.lower()
        digest = hashlib.sha256(file_bytes).hexdigest()
        uploaded_path = SESSION_INPUT_DIR / f"upload_{digest}{extension}"
        if not uploaded_path.exists():
            uploaded_path.write_bytes(file_bytes)
            st.success(f"Saved: {uploaded_file.name}")

    st.markdown(
        "<p style='text-align: center; color: gray;'>- OR -</p>",
        unsafe_allow_html=True,
    )
    video_url = st.text_input("🔗 Paste a video link:")
    if st.button("📥 Download from Link", use_container_width=True) and video_url:
        with st.spinner("Downloading video from link... Please wait."):
            try:
                download_video_from_url(video_url, SESSION_INPUT_DIR)
                st.success("Download complete! Select it from storage below.")
            except (ValueError, RuntimeError) as error:
                st.error(str(error))

    st.divider()
    st.subheader("📼 Stored Videos Manager")
    existing_videos = sorted(
        path.name
        for path in SESSION_INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    selected_video = None
    process_btn = False
    if existing_videos:
        selected_video = st.selectbox(
            "Select video for Q&A:",
            existing_videos,
            key="video_selector",
        )
        if selected_video != st.session_state.last_selected_video:
            st.session_state.last_selected_video = selected_video
            st.session_state.current_video = None
            st.session_state.current_video_path = None
            st.session_state.current_video_id = None
            st.session_state.messages = []

        process_col, delete_col = st.columns(2)
        with process_col:
            process_btn = st.button(
                "🚀 Process",
                type="primary",
                use_container_width=True,
            )
        with delete_col:
            delete_btn = st.button(
                "🗑️ Delete",
                type="secondary",
                use_container_width=True,
            )

        if delete_btn:
            video_path = str(SESSION_INPUT_DIR / selected_video)
            video_id = st.session_state.video_ids.get(video_path)
            try:
                if video_id:
                    get_chroma_collection(str(DB_DIR)).delete(
                        where={
                            "$and": [
                                {"session_id": st.session_state.session_id},
                                {"video_id": video_id},
                            ]
                        }
                    )
                (SESSION_INPUT_DIR / selected_video).unlink()
                st.session_state.video_ids.pop(video_path, None)
                if st.session_state.current_video_path == video_path:
                    st.session_state.current_video = None
                    st.session_state.current_video_path = None
                    st.session_state.current_video_id = None
                    st.session_state.messages = []
                st.success(f"Deleted {selected_video}.")
                st.rerun()
            except Exception as error:
                st.error(f"Could not delete the selected video: {error}")
    else:
        st.info("No videos found. Upload or download one.")

st.markdown(
    '<p class="main-title">🎓 AI Teaching Assistant</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Your intelligent video lecture companion powered by RAG & Groq LLM</p>',
    unsafe_allow_html=True,
)
st.divider()

if st.session_state.current_video:
    st.info(f"🟢 **Currently Active Video for Q&A:** `{st.session_state.current_video}`")

if process_btn and selected_video:
    video_path = str(SESSION_INPUT_DIR / selected_video)
    video_id = st.session_state.video_ids.setdefault(
        video_path,
        str(uuid.uuid4()),
    )
    session_id = st.session_state.session_id

    st.session_state.current_video = None
    st.session_state.current_video_path = None
    st.session_state.current_video_id = None
    st.session_state.messages = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text("🎙️ Step 1/4: Extracting audio...")
        audio_path = str(OUTPUT_DIR / f"{video_id}.mp3")
        audio_file = convert_video_to_mp3(video_path, audio_path)
        progress_bar.progress(25)

        status_text.text("📝 Step 2/4: Transcribing video...")
        json_path = str(OUTPUT_DIR / f"{video_id}.json")
        chunks = transcribe_and_optimize_chunks(audio_file, json_path)
        progress_bar.progress(50)

        status_text.text("🎓 Step 3/4: Checking educational content...")
        educational, reason = classify_educational_content(chunks, selected_video)
        if not educational:
            raise ValueError(NON_EDUCATIONAL_MESSAGE)
        progress_bar.progress(75)

        status_text.text("🗄️ Step 4/4: Indexing transcript for this video...")
        store_chunks_in_chroma(
            chunks,
            db_path=str(DB_DIR),
            video_id=video_id,
            session_id=session_id,
        )

        st.session_state.current_video = selected_video
        st.session_state.current_video_path = video_path
        st.session_state.current_video_id = video_id
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        st.success(f"🎉 '{selected_video}' successfully processed and ready for questions.")
    except Exception as error:
        progress_bar.empty()
        status_text.empty()
        if str(error) == NON_EDUCATIONAL_MESSAGE:
            st.error(NON_EDUCATIONAL_MESSAGE)
        else:
            st.error(f"❌ Could not process this video: {error}")

if st.session_state.current_video_id:
    st.subheader(f"💬 Chatting about: `{st.session_state.current_video}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question about this video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Retrieving context from this video..."):
                try:
                    collection = get_chroma_collection(str(DB_DIR))
                    chat_history = st.session_state.messages[:-1]
                    prior_questions = [
                        message["content"]
                        for message in chat_history
                        if message["role"] == "user"
                    ][-6:]
                    retrieval_query = " ".join(prior_questions + [prompt])
                    top_chunks = retrieve_top_chunks(
                        retrieval_query,
                        collection,
                        video_id=st.session_state.current_video_id,
                        session_id=st.session_state.session_id,
                        top_k=5,
                    )
                    answer = generate_ai_tutor_response(
                        prompt,
                        top_chunks,
                        chat_history=chat_history,
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    error_message = f"⚠️ Could not answer this question: {error}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
else:
    st.info(
        "👈 Select a video, click **'Process'**, and wait for it to pass the "
        "transcript and educational-content checks."
    )
