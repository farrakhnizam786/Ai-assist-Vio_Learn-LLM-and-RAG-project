# AI Teaching Assistant

An educational video question-answering app. It extracts speech from a video, creates a timestamped
transcript, checks whether the transcript is educational/informative, indexes that transcript in
ChromaDB, and answers questions using retrieved content from that video.

## What the app does

1. Accepts a local video upload or downloads a supported video URL with `yt-dlp`.
2. Extracts audio from the selected video with MoviePy.
3. Transcribes the audio with OpenAI Whisper and groups transcript segments into chunks of about
   400 characters. Chunk timestamps are kept in seconds.
4. Rejects an empty transcript. Transcripts with fewer than 20 words are considered insufficient
   for educational classification.
5. Asks the Groq model to classify the transcript as educational/informative or not. The transcript
   is the main evidence; the filename/title is supplemental only.
6. Embeds each transcript chunk with `all-MiniLM-L6-v2` and stores it in ChromaDB with its
   timestamps, `video_id`, and `session_id`.
7. Retrieves relevant chunks only from the active video and session, then asks Groq to answer from
   those chunks. The answer prompt allows chat history only to clarify follow-up references, not
   as evidence for facts.

If a video is rejected, the app displays:

> This video does not appear to contain educational content. Please provide an educational video to continue.

If there is no retrieved transcript evidence, the app responds:

> The answer to this question was not found in the provided video's content.

## Requirements and setup

- Python 3.10+
- FFmpeg available on `PATH` (used by Whisper)
- A Groq API key

Install dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

For local development, create a `.env` file beside `app.py` (copy `.env.example`):

```text
GROQ_API_KEY=your_groq_api_key
```

Run the Streamlit app:

```powershell
streamlit run app.py
```

Upload a video or enter a supported URL, select the video in the sidebar, and click **Process**.
Wait for processing and educational classification to finish before asking questions.

## Deploy publicly with Streamlit Community Cloud

1. Push this project to a GitHub repository. Make sure `.env`, `.streamlit/secrets.toml`,
   `input_videos/`, `output_data/`, and `vector_db/` are not committed. The `.gitignore` excludes
   them; `.env.example` contains only a placeholder.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and choose **Create
   app**.
3. Select the repository and branch, set **Main file path** to `app.py`, and deploy. If prompted
   for a Python version, choose Python 3.11.
4. In the deployed app's **Settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "your_actual_groq_api_key"
   ```

   Do not put the key in source code, `requirements.txt`, or GitHub. The application reads this
   Streamlit secret in the cloud and reads `GROQ_API_KEY` from `.env`/environment variables locally.
5. Wait for the first build to finish, then open the Streamlit URL it provides. The first run may
   take longer because Whisper and the embedding model must be downloaded.

`packages.txt` installs FFmpeg on the Streamlit host, which Whisper needs. `requirements.txt`
installs the Python packages.

### Hosting limitations

The app's uploaded/downloaded videos are stored in session-specific folders on the server, and its
Chroma database is local to the app. This prevents the UI from listing another session's videos,
but Streamlit Community Cloud local files are not durable storage: files and indexes can disappear
when the app restarts, redeploys, or its instance is replaced. Users may need to upload/process
videos again. A durable multi-user service needs external object storage and a managed vector
database; those services are not configured by this project.

Whisper transcription and embedding generation use substantial CPU, memory, and disk. Large videos
or simultaneous users can exceed free-host resource limits. The deployed link is public unless
access is restricted in the hosting provider's settings; do not upload private videos or secrets.

There is also a local-file command-line workflow:

```powershell
python main.py
```

It asks for a filename inside `input_videos/`, processes that video, and answers one question in
the terminal.

## Video and session isolation

- Each Streamlit browser session gets a new `session_id`. Each video path tracked in that session
  gets a stable `video_id`.
- Uploaded files are saved under a content-hash filename inside a session-specific folder. URL
  downloads receive unique filenames in that same folder.
- Audio and transcript outputs are named with the video ID. They are regenerated when processing
  runs; previous files are not treated as a cache.
- ChromaDB uses a shared persistent collection, but every chunk carries `video_id` and `session_id`.
  Retrieval filters on both IDs. A chunk from a different video or session therefore cannot be
  returned as context for the active chat.
- Reprocessing deletes and replaces only the chunks for the same video/session pair. Other videos
  and sessions are preserved. Older database chunks without these metadata fields remain stored
  but are not returned by the filtered retrieval.
- Selecting a different video clears the active chat and deactivates the previous video until the
  selected one is processed. Follow-up chat history is retained only for the active video.

These are application-level isolation controls in the local Chroma database. They do not replace
database access controls if the database is exposed to untrusted users or services.

## Project files

| File/folder | Purpose |
|---|---|
| `app.py` | Streamlit UI, upload/download, video/session state, processing flow, and chat. |
| `main.py` | Command-line pipeline for one local video and one question. |
| `transcribe.py` | Video audio extraction and fresh Whisper transcription/chunking. |
| `rag_pipeline.py` | Educational classification, embeddings, Chroma indexing/retrieval, and grounded Groq answers. |
| `input_videos/` | Session-specific uploaded and downloaded source video folders (local CLI files may also be placed at the root). |
| `output_data/` | Per-video extracted audio and transcript JSON files. |
| `vector_db/` | Persistent ChromaDB data. |
| `tests/test_video_isolation.py` | Regression tests for video/session isolation, reprocessing, classification boundaries, and empty transcripts. |
| `requirements.txt` | Python package dependencies. |
| `packages.txt` | System packages installed by Streamlit Community Cloud (FFmpeg). |
| `.env.example` | Safe local environment-variable template; contains no real key. |
| `projectF.txt` | Detailed engineering notes and function-by-function project guide. |

## Important functions

- `transcribe.convert_video_to_mp3(video_path, audio_path)` validates the input, replaces any
  existing output audio, extracts the selected video's audio, and reports missing/empty audio.
- `transcribe.transcribe_and_optimize_chunks(audio_path, json_path)` replaces the prior transcript,
  runs Whisper, creates timestamped chunks, and fails if no speech was transcribed.
- `rag_pipeline.classify_educational_content(chunks, video_title)` checks transcript sufficiency
  and asks Groq for a conservative JSON classification.
- `rag_pipeline.store_chunks_in_chroma(chunks, db_path, video_id, session_id)` embeds and upserts
  chunks with video/session/timestamp metadata, replacing only the matching video/session data.
- `rag_pipeline.retrieve_top_chunks(query, collection, video_id, session_id, top_k)` applies the
  Chroma metadata filter before returning context.
- `rag_pipeline.generate_ai_tutor_response(query, top_chunks, chat_history)` answers from retrieved
  text only, includes transcript timestamps where available, and abstains when no evidence is
  supplied.

## Tests

Run the local regression suite:

```powershell
python -m unittest discover -s tests -v
```

The tests mock external Groq, Whisper, and embedding calls where appropriate, and use an ephemeral
Chroma collection to verify that two videos and two sessions stay isolated and that reprocessing
replaces only the intended video's chunks. They do not validate real-world classification accuracy
or run a live video/Groq end-to-end request.
