import json
import logging
import math
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp
from yt_dlp.dependencies import yt_dlp_ejs
from yt_dlp.utils import PostProcessingError


LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


class VideoURLValidationError(ValueError):
    pass


class VideoExtractionError(RuntimeError):
    pass


class VideoDownloadError(RuntimeError):
    pass


class VideoMergeError(RuntimeError):
    pass


class VideoOutputValidationError(RuntimeError):
    pass


def _redact_diagnostic(message):
    return URL_PATTERN.sub("[URL redacted]", str(message))


class _RedactingLogger:
    def debug(self, message):
        LOGGER.debug("yt-dlp: %s", _redact_diagnostic(message))

    def info(self, message):
        LOGGER.info("yt-dlp: %s", _redact_diagnostic(message))

    def warning(self, message):
        LOGGER.warning("yt-dlp: %s", _redact_diagnostic(message))

    def error(self, message):
        LOGGER.error("yt-dlp: %s", _redact_diagnostic(message))


def _failure_reference(stage, error, download_id):
    detail = _redact_diagnostic(error)
    LOGGER.error(
        "YouTube %s failed (download_id=%s, error_type=%s): %s",
        stage,
        download_id,
        type(error).__name__,
        detail,
    )
    return download_id[:8]


def _cleanup_download_files(download_dir, download_id):
    for path in download_dir.glob(f"downloaded_{download_id}.*"):
        try:
            if path.is_file():
                path.unlink()
        except OSError as error:
            LOGGER.warning(
                "Could not remove incomplete download file %s: %s",
                path.name,
                _redact_diagnostic(error),
            )


def _output_candidates(info, download_dir, download_id):
    candidates = []
    filepath = info.get("filepath")
    if filepath:
        candidates.append(Path(filepath))

    candidates.extend(sorted(download_dir.glob(f"downloaded_{download_id}.*")))
    unique_candidates = []
    seen = set()
    directory = download_dir.resolve()
    prefix = f"downloaded_{download_id}."

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if (
            resolved.parent != directory
            or not resolved.name.startswith(prefix)
            or resolved.suffix.lower() not in VIDEO_EXTENSIONS
            or resolved in seen
        ):
            continue
        if not resolved.is_file():
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _validate_output_file(path, download_id):
    try:
        size = path.stat().st_size
    except OSError as error:
        reference = _failure_reference("output validation", error, download_id)
        raise VideoOutputValidationError(
            f"The downloaded video could not be read. Check the app logs (reference {reference})."
        ) from error

    if size == 0:
        message = "yt-dlp produced an empty media file."
        reference = _failure_reference("download", message, download_id)
        raise VideoDownloadError(
            "YouTube returned an empty media response. The server may be temporarily "
            f"blocked or the video may be unavailable. Check the app logs (reference {reference})."
        )

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        message = "ffprobe is not available on the server."
        reference = _failure_reference("output validation", message, download_id)
        raise VideoOutputValidationError(
            "The video was downloaded, but its file could not be validated because "
            f"FFmpeg/ffprobe is missing. Check packages.txt and redeploy (reference {reference})."
        )

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        reference = _failure_reference("output validation", error, download_id)
        raise VideoOutputValidationError(
            "The downloaded video could not be validated with FFprobe. "
            f"Check the app logs (reference {reference})."
        ) from error

    try:
        probe = json.loads(result.stdout)
        duration = float(probe.get("format", {}).get("duration", 0))
        has_video = any(
            stream.get("codec_type") == "video"
            for stream in probe.get("streams", [])
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        duration = 0
        has_video = False

    if result.returncode != 0 or not math.isfinite(duration) or duration <= 0 or not has_video:
        details = result.stderr.strip() or "FFprobe reported invalid media."
        reference = _failure_reference("output validation", details, download_id)
        raise VideoOutputValidationError(
            "The downloaded file is not a valid, playable video. "
            f"Check the app logs (reference {reference})."
        )


def _javascript_runtime_options():
    deno = shutil.which("deno")
    if deno:
        return {"deno": {"path": deno}}, "deno"

    node = shutil.which("node")
    if node:
        return {"node": {"path": node}}, "node"

    return None, "not detected (yt-dlp default: Deno)"


def download_video_from_url(url, download_dir):
    url = url.strip()
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        _ = parsed_url.port
    except ValueError as error:
        raise VideoURLValidationError(
            "Enter a valid HTTP or HTTPS video URL."
        ) from error
    if (
        parsed_url.scheme.lower() not in {"http", "https"}
        or not parsed_url.netloc
        or not hostname
    ):
        raise VideoURLValidationError("Enter a valid HTTP or HTTPS video URL.")

    download_dir = Path(download_dir)
    try:
        download_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VideoDownloadError(
            "The server could not create the video download folder. Check the app logs."
        ) from error

    download_id = uuid.uuid4().hex
    output_template = str(download_dir / f"downloaded_{download_id}.%(ext)s")
    js_runtimes, runtime_name = _javascript_runtime_options()
    ffmpeg_available = bool(shutil.which("ffmpeg"))
    ffprobe_available = bool(shutil.which("ffprobe"))

    LOGGER.info(
        "Starting YouTube download (download_id=%s, yt-dlp=%s, EJS=%s, "
        "JS runtime=%s, ffmpeg=%s, ffprobe=%s)",
        download_id,
        yt_dlp.version.__version__,
        bool(yt_dlp_ejs),
        runtime_name,
        ffmpeg_available,
        ffprobe_available,
    )
    if not yt_dlp_ejs:
        LOGGER.warning(
            "yt-dlp-ejs is not installed; YouTube JavaScript challenge solving may fail. "
            "Install yt-dlp[default]."
        )
    if runtime_name.startswith("not detected"):
        LOGGER.warning("No JavaScript runtime was detected on PATH.")
    if not ffmpeg_available:
        LOGGER.warning("FFmpeg was not detected on PATH; separate streams cannot be merged.")

    options = {
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/"
            "bestvideo+bestaudio/best"
        ),
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "overwrites": True,
        "merge_output_format": "mp4/mkv",
        "logger": _RedactingLogger(),
    }
    if js_runtimes:
        options["js_runtimes"] = js_runtimes

    try:
        downloader_context = yt_dlp.YoutubeDL(options)
    except Exception as error:
        reference = _failure_reference("initialization", error, download_id)
        _cleanup_download_files(download_dir, download_id)
        raise VideoExtractionError(
            "yt-dlp could not initialize. Check the deployed yt-dlp, EJS, and "
            f"JavaScript runtime dependencies (reference {reference})."
        ) from error

    try:
        with downloader_context as downloader:
            try:
                info = downloader.extract_info(url, download=False)
            except Exception as error:
                reference = _failure_reference("extraction", error, download_id)
                detail = str(error).casefold()
                if "403" in detail or "forbidden" in detail:
                    message = (
                        "YouTube rejected the extraction request (HTTP 403). The hosting "
                        "server's IP may be restricted by YouTube; check the app logs "
                        f"(reference {reference})."
                    )
                elif "javascript" in detail or "ejs" in detail or "challenge" in detail:
                    message = (
                        "YouTube's JavaScript challenge could not be solved. Confirm that "
                        "the deployed environment installed yt-dlp's default and Deno "
                        f"dependencies (reference {reference})."
                    )
                else:
                    message = (
                        "Could not extract this video. It may be unavailable, private, "
                        "age-restricted, or temporarily blocked by YouTube. This app does "
                        f"not use account cookies (reference {reference})."
                    )
                raise VideoExtractionError(message) from error

            if not isinstance(info, dict) or not info:
                message = "yt-dlp returned no video information."
                reference = _failure_reference("extraction", message, download_id)
                raise VideoExtractionError(
                    f"Could not extract this video (reference {reference})."
                )

            try:
                downloaded_info = downloader.process_ie_result(info, download=True)
            except PostProcessingError as error:
                reference = _failure_reference("merge", error, download_id)
                raise VideoMergeError(
                    "The video streams downloaded, but FFmpeg could not merge them. "
                    f"Confirm ffmpeg is installed through packages.txt (reference {reference})."
                ) from error
            except Exception as error:
                reference = _failure_reference("download", error, download_id)
                detail = str(error).casefold()
                if "403" in detail or "forbidden" in detail:
                    message = (
                        "YouTube rejected the media request (HTTP 403). The hosting "
                        "server's IP may be restricted by YouTube; check the app logs "
                        f"(reference {reference})."
                    )
                elif "empty" in detail or "zero byte" in detail:
                    message = (
                        "YouTube returned an empty media response. The server may be "
                        "temporarily blocked or the video may be unavailable; check the "
                        f"app logs (reference {reference})."
                    )
                else:
                    message = (
                        "The video was extracted, but downloading its media failed. "
                        f"Check the app logs (reference {reference})."
                    )
                raise VideoDownloadError(message) from error

            candidates = _output_candidates(
                downloaded_info or {},
                download_dir,
                download_id,
            )
            if not candidates:
                message = "yt-dlp completed without producing a video file."
                reference = _failure_reference("download", message, download_id)
                raise VideoDownloadError(
                    f"No video file was produced (reference {reference})."
                )

            output_path = candidates[0]
            _validate_output_file(output_path, download_id)
            LOGGER.info(
                "YouTube download completed (download_id=%s, size_bytes=%s)",
                download_id,
                output_path.stat().st_size,
            )
            return str(output_path)
    except (
        VideoExtractionError,
        VideoDownloadError,
        VideoMergeError,
        VideoOutputValidationError,
    ):
        _cleanup_download_files(download_dir, download_id)
        raise
