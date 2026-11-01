import json
import os
import tempfile
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chromadb
import rag_pipeline
import transcribe


class EmbeddingResult:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeEmbeddingModel:
    def encode(self, value):
        if isinstance(value, str):
            return EmbeddingResult([1.0, 0.0])
        return EmbeddingResult([[1.0, 0.0] for _ in value])


class VideoIsolationTests(unittest.TestCase):
    def test_groq_client_reads_streamlit_cloud_secret(self):
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": ""}),
            patch.object(
                rag_pipeline.st,
                "secrets",
                {"GROQ_API_KEY": "streamlit-cloud-key"},
            ),
            patch.object(rag_pipeline, "Groq") as groq,
        ):
            rag_pipeline._get_groq_client()

        groq.assert_called_once_with(api_key="streamlit-cloud-key")

    def test_indexing_upserts_chunks_with_video_and_session_metadata(self):
        collection = Mock()
        chunks = [
            {"start": 1.0, "end": 3.0, "text": "First lesson chunk"},
            {"start": 3.0, "end": 6.0, "text": "Second lesson chunk"},
        ]
        with (
            patch.object(rag_pipeline, "get_chroma_collection", return_value=collection),
            patch.object(rag_pipeline, "_get_embedding_model", return_value=FakeEmbeddingModel()),
        ):
            result = rag_pipeline.store_chunks_in_chroma(
                chunks,
                db_path="test-db",
                video_id="video-b",
                session_id="session-1",
            )

        self.assertIs(result, collection)
        collection.delete.assert_called_once_with(
            where={
                "$and": [
                    {"session_id": "session-1"},
                    {"video_id": "video-b"},
                ]
            }
        )
        collection.upsert.assert_called_once()
        indexed = collection.upsert.call_args.kwargs
        self.assertEqual(indexed["ids"], ["session-1:video-b:0", "session-1:video-b:1"])
        self.assertEqual(
            indexed["metadatas"],
            [
                {
                    "session_id": "session-1",
                    "video_id": "video-b",
                    "start": 1.0,
                    "end": 3.0,
                },
                {
                    "session_id": "session-1",
                    "video_id": "video-b",
                    "start": 3.0,
                    "end": 6.0,
                },
            ],
        )

    def test_retrieval_filters_by_active_video_and_session(self):
        collection = Mock()
        collection.query.return_value = {
            "documents": [["The second video explains recursion."]],
            "metadatas": [[{"start": 12.0, "end": 18.0}]],
        }
        with patch.object(
            rag_pipeline,
            "_get_embedding_model",
            return_value=FakeEmbeddingModel(),
        ):
            chunks = rag_pipeline.retrieve_top_chunks(
                "How does recursion work?",
                collection,
                video_id="video-b",
                session_id="session-2",
                top_k=5,
            )

        self.assertEqual(chunks[0]["text"], "The second video explains recursion.")
        collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 0.0]],
            n_results=5,
            where={
                "$and": [
                    {"session_id": "session-2"},
                    {"video_id": "video-b"},
                ]
            },
        )

    def test_chroma_isolates_videos_and_replaces_only_the_reprocessed_video(self):
        client = chromadb.EphemeralClient()
        collection = client.create_collection(name=f"video_lectures_{uuid.uuid4().hex}")
        chunks = [{"start": 0.0, "end": 2.0, "text": "first transcript"}]
        with (
            patch.object(rag_pipeline, "get_chroma_collection", return_value=collection),
            patch.object(
                rag_pipeline,
                "_get_embedding_model",
                return_value=FakeEmbeddingModel(),
            ),
        ):
            rag_pipeline.store_chunks_in_chroma(chunks, "unused", "video-a", "session-1")
            rag_pipeline.store_chunks_in_chroma(
                [{"start": 0.0, "end": 2.0, "text": "second transcript"}],
                "unused",
                "video-b",
                "session-1",
            )
            rag_pipeline.store_chunks_in_chroma(
                [{"start": 0.0, "end": 2.0, "text": "another user's transcript"}],
                "unused",
                "video-b",
                "session-2",
            )
            rag_pipeline.store_chunks_in_chroma(
                [{"start": 0.0, "end": 2.0, "text": "updated second transcript"}],
                "unused",
                "video-b",
                "session-1",
            )
            active_chunks = rag_pipeline.retrieve_top_chunks(
                "What is in this video?",
                collection,
                video_id="video-b",
                session_id="session-1",
            )

        self.assertEqual(collection.count(), 3)
        self.assertEqual(
            [chunk["text"] for chunk in active_chunks],
            ["updated second transcript"],
        )

    def test_classifier_uses_transcript_and_rejects_insufficient_text(self):
        transcript = (
            "The instructor demonstrates recursion with a base case and a recursive "
            "case, then explains how each call reduces the remaining problem."
        )
        message = SimpleNamespace(
            content=json.dumps({"educational": True, "reason": "A programming lesson."})
        )
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        with patch.object(rag_pipeline, "_get_groq_client", return_value=client):
            result = rag_pipeline.classify_educational_content(
                [{"text": transcript}],
                video_title="Unrelated promotional title",
            )

        self.assertTrue(result[0])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn(transcript, prompt)
        self.assertIn("must never determine", prompt)
        with patch.object(rag_pipeline, "_get_groq_client") as groq_client:
            with self.assertRaisesRegex(ValueError, "Please provide another video"):
                rag_pipeline.classify_educational_content([{"text": "Too short"}])
        groq_client.assert_not_called()

    def test_no_retrieved_context_returns_required_abstention_without_llm_call(self):
        with patch.object(rag_pipeline, "_get_groq_client") as groq_client:
            answer = rag_pipeline.generate_ai_tutor_response("Question?", [])
        self.assertEqual(answer, rag_pipeline.ANSWER_NOT_FOUND)
        groq_client.assert_not_called()

    def test_generation_prompt_limits_facts_to_current_transcript(self):
        message = SimpleNamespace(content="Recursion calls itself.")
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        with patch.object(rag_pipeline, "_get_groq_client", return_value=client):
            answer = rag_pipeline.generate_ai_tutor_response(
                "Why is the base case needed?",
                [{"text": "A base case stops recursion.", "start": 2, "end": 6}],
                chat_history=[{"role": "user", "content": "Explain recursion."}],
            )

        self.assertEqual(answer, "Recursion calls itself.")
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn(rag_pipeline.ANSWER_NOT_FOUND, prompt)
        self.assertIn("A base case stops recursion.", prompt)
        self.assertIn("user: Explain recursion.", prompt)
        self.assertNotIn("previous video transcript", prompt)


class TranscriptionTests(unittest.TestCase):
    def test_existing_audio_is_replaced_from_the_selected_video(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "video.mp4")
            audio_path = os.path.join(temp_dir, "audio.mp3")
            with open(video_path, "wb") as video_file:
                video_file.write(b"video")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"stale audio")

            def write_fresh_audio(path, codec):
                with open(path, "wb") as audio_file:
                    audio_file.write(b"fresh audio")

            audio = SimpleNamespace(
                write_audiofile=Mock(side_effect=write_fresh_audio)
            )
            clip = SimpleNamespace(audio=audio, close=Mock())
            with patch.object(transcribe, "VideoFileClip", return_value=clip):
                result = transcribe.convert_video_to_mp3(video_path, audio_path)

            self.assertEqual(result, audio_path)
            self.assertEqual(audio.write_audiofile.call_count, 1)
            self.assertEqual(clip.close.call_count, 1)
            with open(audio_path, "rb") as audio_file:
                self.assertEqual(audio_file.read(), b"fresh audio")

    def test_existing_transcript_is_replaced_with_fresh_transcription(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.mp3")
            json_path = os.path.join(temp_dir, "transcript.json")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(json_path, "w", encoding="utf-8") as transcript_file:
                transcript_file.write('[{"text": "stale transcript"}]')

            model = Mock()
            model.transcribe.return_value = {
                "segments": [{"start": 2.0, "end": 4.0, "text": "Fresh transcript"}]
            }
            with patch.object(transcribe.whisper, "load_model", return_value=model):
                chunks = transcribe.transcribe_and_optimize_chunks(audio_path, json_path)

            self.assertEqual(chunks[0]["text"], "Fresh transcript")
            with open(json_path, encoding="utf-8") as transcript_file:
                self.assertEqual(json.load(transcript_file), chunks)

    def test_empty_transcript_fails_instead_of_reusing_previous_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.mp3")
            json_path = os.path.join(temp_dir, "transcript.json")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(json_path, "w", encoding="utf-8") as transcript_file:
                transcript_file.write('[{"text": "old transcript"}]')

            model = Mock()
            model.transcribe.return_value = {"segments": []}
            with patch.object(transcribe.whisper, "load_model", return_value=model):
                with self.assertRaisesRegex(ValueError, "No speech could be transcribed"):
                    transcribe.transcribe_and_optimize_chunks(audio_path, json_path)

            self.assertFalse(os.path.exists(json_path))


if __name__ == "__main__":
    unittest.main()
