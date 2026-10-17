# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Microphone capture and speech recognition.

Audio is recorded in short frames. A calibrated energy detector decides when
a phrase starts and ends. Faster-Whisper then transcribes the completed
phrase. The Whisper model is loaded once and reused.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Set before Faster-Whisper imports Hugging Face so Windows cache warnings stay quiet.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np

# Faster-Whisper expects mono 16 kHz audio. Capturing at this rate avoids a
# resample on most microphones. Devices that refuse 16 kHz are resampled later.
TARGET_SAMPLE_RATE = 16_000

# Repository sizes reported by the Hugging Face API for the Systran
# faster-whisper models. These include the model weights and tokenizer files.
ASR_DOWNLOAD_BYTES = {
    "tiny": 75_538_270,
    "base": 145_217_532,
    "small": 483_546_902,
    "medium": 1_527_906_378,
    "large-v3": 3_145_762_253,
}
ASR_MODEL_SIZES = tuple(ASR_DOWNLOAD_BYTES)
DEFAULT_ASR_MODEL = "base"
SUPPORTED_ASR_LANGUAGES = ("en", "es")


class AudioCaptureError(RuntimeError):
    """The microphone could not be opened or read."""


@dataclass
class DetectorConfig:
    """Settings for the energy-based phrase detector."""

    sample_rate: int = TARGET_SAMPLE_RATE
    threshold: float = 0.015
    silence_seconds: float = 0.8
    min_speech_seconds: float = 0.4
    max_speech_seconds: float = 20.0
    pre_roll_seconds: float = 0.25


def rms(samples: np.ndarray) -> float:
    """Return the root-mean-square energy of a mono audio frame."""

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def threshold_from_noise(
    noise_rms: float,
    floor: float = 0.012,
    multiplier: float = 3.5,
    ceiling: float = 0.08,
) -> float:
    """Choose a speech threshold from a short ambient-noise measurement."""

    return float(min(ceiling, max(floor, noise_rms * multiplier)))


def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Linearly resample mono audio. Whisper needs 16 kHz input."""

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0 or original_rate == target_rate:
        return samples
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    target_length = max(1, int(round(samples.size * target_rate / original_rate)))
    source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def clean_transcript(text: str) -> str:
    """Drop blank transcripts and strings that contain no letters or digits."""

    cleaned = " ".join(text.split())
    if not any(character.isalnum() for character in cleaned):
        return ""
    return cleaned


def to_mono(frame: np.ndarray) -> np.ndarray:
    """Average multi-channel input down to one channel."""

    audio = np.asarray(frame, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    raise ValueError("Audio frame must be one- or two-dimensional.")


class EnergyUtteranceDetector:
    """Find one spoken phrase using audio energy.

    This is intentionally simple: speech starts when the frame energy crosses
    a threshold, and it ends after enough silence or at the maximum duration.
    Very short noises are discarded.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.completed_audio: np.ndarray | None = None
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._speech: list[np.ndarray] = []
        self._speech_samples = 0
        self._loud_samples = 0
        self._silence_samples = 0
        self._active = False

    def reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._speech = []
        self._speech_samples = 0
        self._loud_samples = 0
        self._silence_samples = 0
        self._active = False

    def process_frame(self, frame: np.ndarray) -> str:
        """Consume one mono frame.

        Returns ``listening``, ``recording``, ``finished``, or ``discarded``.
        After ``finished``, ``take_audio`` returns the utterance.
        """

        audio = np.asarray(frame, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return "recording" if self._active else "listening"

        energy = rms(audio)
        if not self._active:
            if energy < self.config.threshold:
                self._push_pre_roll(audio)
                return "listening"
            # Keep a little audio from just before the threshold crossing so the
            # first consonant is not clipped, then start the phrase on this frame.
            self._active = True
            self._speech = [chunk.copy() for chunk in self._pre_roll]
            self._speech.append(audio.copy())
            self._speech_samples = sum(chunk.size for chunk in self._speech)
            self._loud_samples = audio.size
            self._silence_samples = 0
            self._pre_roll.clear()
            self._pre_roll_samples = 0
            return "recording"

        self._speech.append(audio)
        self._speech_samples += audio.size
        if energy >= self.config.threshold:
            self._loud_samples += audio.size
            self._silence_samples = 0
        else:
            self._silence_samples += audio.size

        max_samples = int(self.config.max_speech_seconds * self.config.sample_rate)
        silence_limit = int(self.config.silence_seconds * self.config.sample_rate)
        reached_limit = self._speech_samples >= max_samples
        ended = self._silence_samples >= silence_limit
        if not reached_limit and not ended:
            return "recording"
        return self._finish()

    def take_audio(self) -> np.ndarray | None:
        audio = self.completed_audio
        self.completed_audio = None
        return audio

    def _push_pre_roll(self, audio: np.ndarray) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_samples += audio.size
        limit = int(self.config.pre_roll_seconds * self.config.sample_rate)
        while self._pre_roll and self._pre_roll_samples > limit:
            removed = self._pre_roll.popleft()
            self._pre_roll_samples -= removed.size

    def _finish(self) -> str:
        min_samples = int(self.config.min_speech_seconds * self.config.sample_rate)
        if self._speech and self._loud_samples >= min_samples:
            self.completed_audio = np.concatenate(self._speech).astype(np.float32)
            self.reset()
            return "finished"
        self.completed_audio = None
        self.reset()
        return "discarded"


class MicrophoneUtteranceRecorder:
    """Record one phrase at a time from a selected input device."""

    def __init__(
        self,
        input_device: int | None = None,
        config: DetectorConfig | None = None,
        frame_ms: int = 30,
    ) -> None:
        self.input_device = input_device
        self.detector = EnergyUtteranceDetector(config or DetectorConfig())
        self.frame_ms = frame_ms

    def calibrate(self, seconds: float = 0.6) -> float:
        """Measure ambient noise and set the speech threshold."""

        noise = self._record_for(seconds)
        chosen = threshold_from_noise(rms(noise))
        self.detector.config.threshold = chosen
        return chosen

    def record_utterance(self) -> np.ndarray | None:
        """Block until one usable phrase ends.

        Returns 16 kHz mono audio, or ``None`` when the sound was too short
        or too quiet to transcribe.
        """

        import sounddevice as sd

        stream, capture_rate = self._open_stream()
        frame_samples = max(1, int(capture_rate * self.frame_ms / 1000))
        try:
            while True:
                frame, _overflowed = stream.read(frame_samples)
                mono = to_mono(frame)
                if capture_rate != TARGET_SAMPLE_RATE:
                    mono = resample_audio(mono, capture_rate, TARGET_SAMPLE_RATE)
                status = self.detector.process_frame(mono)
                if status == "finished":
                    return self.detector.take_audio()
                if status == "discarded":
                    return None
        finally:
            stream.close()
            sd.stop()

    def _record_for(self, seconds: float) -> np.ndarray:
        import sounddevice as sd

        stream, capture_rate = self._open_stream()
        total_frames = max(1, int(capture_rate * seconds))
        try:
            frame, _overflowed = stream.read(total_frames)
        finally:
            stream.close()
        mono = to_mono(frame)
        if capture_rate != TARGET_SAMPLE_RATE:
            mono = resample_audio(mono, capture_rate, TARGET_SAMPLE_RATE)
        return mono

    def _open_stream(self):
        import sounddevice as sd

        attempts: list[tuple[int | None, int]] = [
            (TARGET_SAMPLE_RATE, 1),
            (TARGET_SAMPLE_RATE, 2),
            (None, 1),
            (None, 2),
        ]
        errors: list[str] = []
        for sample_rate, channels in attempts:
            try:
                kwargs = {
                    "device": self.input_device,
                    "channels": channels,
                    "dtype": "float32",
                }
                if sample_rate is not None:
                    kwargs["samplerate"] = sample_rate
                stream = sd.InputStream(**kwargs)
            except Exception as exc:  # PortAudio raises several exception types.
                errors.append(str(exc))
                continue
            try:
                stream.start()
            except Exception as exc:
                stream.close()
                errors.append(str(exc))
                continue
            return stream, int(stream.samplerate)

        detail = errors[-1] if errors else "No audio backend error was returned."
        raise AudioCaptureError(
            "Could not open the microphone. Check Windows microphone permission, "
            "confirm a device with --list-devices, and reinstall sounddevice if "
            f"PortAudio failed. Last error: {detail}"
        )


class SpeechRecognizer:
    """Transcribe completed utterances with one cached Faster-Whisper model."""

    def __init__(
        self,
        model_size: str = DEFAULT_ASR_MODEL,
        models_dir: Path | str = "models/asr",
        device: str = "cpu",
    ) -> None:
        if model_size not in ASR_DOWNLOAD_BYTES:
            known = ", ".join(ASR_MODEL_SIZES)
            raise ValueError(f"Unknown Whisper model '{model_size}'. Choose one of: {known}.")
        if device not in {"cpu", "cuda"}:
            raise ValueError("ASR device must be 'cpu' or 'cuda'.")
        self.model_size = model_size
        self.models_dir = Path(models_dir)
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the Whisper model, downloading it on first use."""

        if self._model is not None:
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AudioCaptureError(
                "faster-whisper is not installed. Install requirements before running."
            ) from exc

        compute_type = "int8" if self.device == "cpu" else "float16"
        print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        try:
            # download_root keeps the CTranslate2 weights inside this project.
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
                download_root=str(self.models_dir),
            )
        except Exception as exc:
            raise AudioCaptureError(
                "Could not download or load the Whisper model. An internet connection "
                "is required the first time. If a download was interrupted, run the "
                f"download command again. Details: {exc}"
            ) from exc

    def transcribe(self, audio: np.ndarray | None, language: str) -> str:
        """Return recognized text, or an empty string for silence."""

        if language not in SUPPORTED_ASR_LANGUAGES:
            raise ValueError(
                f"Unsupported ASR language '{language}'. "
                f"Choose one of: {', '.join(SUPPORTED_ASR_LANGUAGES)}."
            )
        if audio is None:
            return ""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        # A silent or empty buffer has nothing useful for Whisper to read.
        if samples.size < TARGET_SAMPLE_RATE * 0.2 or rms(samples) < 1e-4:
            return ""
        self.load()
        assert self._model is not None
        try:
            segments, _info = self._model.transcribe(
                samples,
                language=language,
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            text = " ".join(segment.text.strip() for segment in segments)
        except Exception as exc:
            raise AudioCaptureError(f"Speech recognition failed: {exc}") from exc
        return clean_transcript(text)
