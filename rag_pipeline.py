import json
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

ANSWER_NOT_FOUND = "The answer to this question was not found in the provided video's content."
NON_EDUCATIONAL_MESSAGE = (
    "This video does not appear to contain educational content. "
    "Please provide an educational video to continue."
)
LLM_MODEL = "openai/gpt-oss-120b"
MIN_TRANSCRIPT_WORDS = 20


def format_time(seconds):
    """Convert seconds to MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except StreamlitSecretNotFoundError:
            api_key = None
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set it in .env for local use or "
            "in Streamlit Cloud app secrets for deployment."
        )
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def _get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_chroma_collection(db_path="vector_db"):
    """Return the persistent collection used for all isolated video records."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name="video_lectures")


def _chunk_text(chunks):
    return " ".join(
        chunk.get("text", "").strip()
        for chunk in chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    )


def classify_educational_content(chunks, video_title=""):
    """Classify transcript content; a title alone is never sufficient evidence."""
    transcript = _chunk_text(chunks)
    if len(transcript.split()) < MIN_TRANSCRIPT_WORDS:
        raise ValueError(
            "The video transcript is empty or too short to classify. "
            "Please provide another video with clear spoken content."
        )

    evidence_text = transcript[:12000]
    prompt = f"""
Classify whether the following video's spoken content is educational or genuinely informative.
Accept lessons, lectures, tutorials, explanations, technical discussions, and informative
documentaries. Reject movies, entertainment shorts, music videos, advertisements, and company
promotions that only market a product or organization. Technical vocabulary by itself is not
evidence of education: the transcript must actually teach, explain, or inform. Use the transcript
as the primary evidence; the title is only supplemental and must never determine the result.
Return only a JSON object with boolean "educational" and a concise string "reason".

Title (supplemental only): {video_title}
Transcript:
{evidence_text}
"""
    response = _get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Classify video content conservatively using transcript evidence only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The content classifier returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError("The content classifier returned invalid JSON.") from error

    if not isinstance(result, dict):
        raise RuntimeError("The content classifier returned an invalid result.")
    educational = result.get("educational")
    reason = result.get("reason")
    if not isinstance(educational, bool) or not isinstance(reason, str):
        raise RuntimeError("The content classifier returned an invalid result.")
    return educational, reason


def store_chunks_in_chroma(chunks, db_path, video_id, session_id):
    """Replace only this session/video's chunks and store isolated metadata."""
    if not video_id or not session_id:
        raise ValueError("Both video_id and session_id are required for indexing.")

    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]
    if not valid_chunks:
        raise ValueError("Cannot index a video without transcript text.")

    collection = get_chroma_collection(db_path)
    where = {"$and": [{"session_id": session_id}, {"video_id": video_id}]}
    collection.delete(where=where)

    ids = []
    documents = []
    metadatas = []
    for index, chunk in enumerate(valid_chunks):
        ids.append(f"{session_id}:{video_id}:{index}")
        documents.append(chunk["text"].strip())
        metadatas.append(
            {
                "session_id": session_id,
                "video_id": video_id,
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
            }
        )

    embeddings = _get_embedding_model().encode(documents).tolist()
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


def retrieve_top_chunks(query, collection, video_id, session_id, top_k=5):
    """Retrieve transcript chunks only from the active session and video."""
    if not video_id or not session_id:
        raise ValueError("Both video_id and session_id are required for retrieval.")

    query_embedding = _get_embedding_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [{"session_id": session_id}, {"video_id": video_id}]},
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    return [
        {
            "text": document,
            "start": metadata["start"],
            "end": metadata["end"],
        }
        for document, metadata in zip(documents, metadatas)
        if document and metadata
    ]


def generate_ai_tutor_response(query, top_chunks, chat_history=None):
    """Generate an answer grounded exclusively in the retrieved video transcript."""
    if not top_chunks:
        return ANSWER_NOT_FOUND

    context_lines = [
        f"[Timestamp: {format_time(chunk['start'])} - {format_time(chunk['end'])}] "
        f"{chunk['text']}"
        for chunk in top_chunks
    ]
    history_lines = []
    for message in (chat_history or [])[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history_lines.append(f"{role}: {content}")

    prompt = f"""
Answer the student's question using only the video transcript evidence below.
Conversation history is included only to resolve references in a follow-up question; it is not
evidence and must not supply facts. Do not use outside knowledge or invent details, examples,
timestamps, or explanations. If the answer is not supported by the transcript evidence, reply
exactly: "{ANSWER_NOT_FOUND}"
Include timestamp ranges from the transcript when they support the answer.

Same-video conversation history:
{chr(10).join(history_lines) or "(none)"}

Current video transcript evidence:
{chr(10).join(context_lines)}

Student question:
{query}
"""
    response = _get_groq_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video-grounded educational assistant. The provided transcript "
                    "is the only source of factual answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The question-answering model returned an empty response.")
    return answer.strip()
