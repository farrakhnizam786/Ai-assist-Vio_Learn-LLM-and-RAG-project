import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import transcribe
import video_downloader
from yt_dlp.utils import DownloadError, PostProcessingError


class FakeYoutubeDL:
    def __init__(
        self,
        options,
        output_bytes=b"video bytes",
        extract_error=None,
        process_error=None,
    ):
        self.options = options
        self.output_bytes = output_bytes
        self.extract_error = extract_error
        self.process_error = process_error
        self.output_path = Path(
            options["outtmpl"].replace("%(ext)s", "mp4")
        )
        self.processed = False

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def extract_info(self, url, download):
        self.url = url
        self.extract_download = download
        if self.extract_error:
            raise self.extract_error
        return {
            "id": "test-video",
            "ext": "mp4",
            "requested_formats": [
                {"format_id": "video-only"},
                {"format_id": "audio-only"},
            ],
        }

    def process_ie_result(self, info, download):
        self.processed = download
        self.output_path.write_bytes(self.output_bytes)
        if self.process_error:
            raise self.process_error
        return {**info, "filepath": str(self.output_path)}


class VideoDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.options = []
        self.downloaders = []

    @staticmethod
    def _which(name):
        return {
            "deno": "C:/tools/deno.exe",
            "ffmpeg": "C:/tools/ffmpeg.exe",
            "ffprobe": "C:/tools/ffprobe.exe",
        }.get(name)

    @staticmethod
    def _valid_probe():
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "12.5"},
                    "streams": [{"codec_type": "video"}],
                }
            ),
            stderr="",
        )

    def _download(
        self,
        url="https://www.youtube.com/watch?v=video-id",
        output_bytes=b"video bytes",
        extract_error=None,
        process_error=None,
        probe_result=None,
    ):
        def create_downloader(options):
            self.options.append(options)
            downloader = FakeYoutubeDL(
                options,
                output_bytes=output_bytes,
                extract_error=extract_error,
                process_error=process_error,
            )
            self.downloaders.append(downloader)
            return downloader

        with (
            patch.object(
                video_downloader.yt_dlp,
                "YoutubeDL",
                side_effect=create_downloader,
            ),
            patch.object(
                video_downloader.shutil,
                "which",
                side_effect=self._which,
            ),
            patch.object(
                video_downloader.subprocess,
                "run",
                return_value=probe_result or self._valid_probe(),
            ),
        ):
            return video_downloader.download_video_from_url(url, self.temp_dir.name)

    def test_standard_shortened_and_shorts_urls_are_accepted(self):
        urls = [
            "https://www.youtube.com/watch?v=video-id",
            "https://youtu.be/video-id",
            "https://www.youtube.com/shorts/video-id",
        ]
        paths = []

        for url in urls:
            with self.subTest(url=url):
                result = self._download(url)
                paths.append(result)
                self.assertTrue(Path(result).is_file())
                self.assertEqual(self.downloaders[-1].url, url)
                self.assertFalse(self.downloaders[-1].extract_download)

        self.assertEqual(len(set(paths)), len(urls))
    def test_invalid_url_is_rejected_before_extraction(self):
        for url in (
            "not a URL",
            "ftp://youtube.com/watch?v=video-id",
            "https://[",
            "https://youtube.com:invalid/watch?v=video-id",
        ):
            with self.subTest(url=url):
                with self.assertRaises(video_downloader.VideoURLValidationError):
                    self._download(url)

        self.assertEqual(self.options, [])

    def test_separate_streams_are_supported_with_ffmpeg_merging(self):
        self._download()
        options = self.options[-1]
        self.assertIn("bestvideo[ext=mp4]+bestaudio[ext=m4a]", options["format"])
        self.assertIn("bestvideo+bestaudio/best", options["format"])
        self.assertEqual(options["merge_output_format"], "mp4/mkv")
        self.assertEqual(options["js_runtimes"], {"deno": {"path": "C:/tools/deno.exe"}})
        self.assertTrue(self.downloaders[-1].processed)

    def test_unavailable_private_video_is_an_extraction_error(self):
        with self.assertRaises(video_downloader.VideoExtractionError) as raised:
            self._download(extract_error=DownloadError("Video unavailable: private video"))

        self.assertIn("private", str(raised.exception))
        self.assertNotIn("valid HTTP", str(raised.exception))
        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_empty_output_is_rejected_and_removed(self):
        with self.assertRaisesRegex(video_downloader.VideoDownloadError, "empty media"):
            self._download(output_bytes=b"")

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_empty_download_error_is_distinguished_and_removed(self):
        with self.assertRaisesRegex(video_downloader.VideoDownloadError, "empty media"):
            self._download(
                output_bytes=b"partial",
                process_error=DownloadError("The downloaded file is empty"),
            )

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_http_403_is_reported_as_a_download_failure(self):
        with self.assertRaisesRegex(video_downloader.VideoDownloadError, "HTTP 403"):
            self._download(process_error=DownloadError("HTTP Error 403: Forbidden"))

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_merge_failure_is_distinguished_and_partial_file_removed(self):
        with self.assertRaises(video_downloader.VideoMergeError):
            self._download(process_error=PostProcessingError("ffmpeg merge failed"))

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_corrupted_output_is_rejected_and_removed(self):
        invalid_probe = SimpleNamespace(
            returncode=1,
            stdout="{}",
            stderr="Invalid data found when processing input",
        )
        with self.assertRaises(video_downloader.VideoOutputValidationError):
            self._download(probe_result=invalid_probe)

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_non_finite_media_duration_is_rejected(self):
        invalid_probe = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "nan"},
                    "streams": [{"codec_type": "video"}],
                }
            ),
            stderr="",
        )
        with self.assertRaises(video_downloader.VideoOutputValidationError):
            self._download(probe_result=invalid_probe)

        self.assertEqual(list(Path(self.temp_dir.name).iterdir()), [])

    def test_successful_download_proceeds_to_transcription(self):
        video_path = self._download()
        audio_path = Path(self.temp_dir.name) / "audio.mp3"
        json_path = Path(self.temp_dir.name) / "transcript.json"

        def write_audio(path, codec):
            Path(path).write_bytes(b"fresh audio")

        clip = SimpleNamespace(
            audio=SimpleNamespace(write_audiofile=Mock(side_effect=write_audio)),
            close=Mock(),
        )
        model = Mock()
        model.transcribe.return_value = {
            "segments": [
                {
                    "start": 1.0,
                    "end": 4.0,
                    "text": "A spoken lesson from the downloaded video.",
                }
            ]
        }
        with (
            patch.object(transcribe, "VideoFileClip", return_value=clip),
            patch.object(transcribe.whisper, "load_model", return_value=model),
        ):
            extracted_audio = transcribe.convert_video_to_mp3(
                video_path,
                str(audio_path),
            )
            chunks = transcribe.transcribe_and_optimize_chunks(
                extracted_audio,
                str(json_path),
            )

        self.assertTrue(audio_path.is_file())
        self.assertTrue(json_path.is_file())
        self.assertEqual(chunks[0]["text"], "A spoken lesson from the downloaded video.")


if __name__ == "__main__":
    unittest.main()
