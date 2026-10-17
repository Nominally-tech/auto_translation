# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local text-to-speech with Piper voices.

Voices are downloaded with Piper's own ``download_voice`` helper into
``models/tts``. A voice is loaded once and reused. Playback uses the sample
rate and channel count from the generated audio instead of assuming 16 kHz.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Verified against the rhasspy/piper-voices repository. Sizes are the .onnx
# files only; each voice also has a small .onnx.json config.
# A finished medium Piper voice is about 63 MB. Anything well below the
# published size is treated as an interrupted download.


@dataclass(frozen=True)
class VoiceInfo:
    voice_id: str
    language: str
    language_name: str
    sample_rate: int
    size_bytes: int
    license_summary: str
    dataset_url: str
    model_card_url: str


VOICES = {
    "en_US-ljspeech-medium": VoiceInfo(
        voice_id="en_US-ljspeech-medium",
        language="en",
        language_name="English",
        sample_rate=22_050,
        size_bytes=63_531_379,
        license_summary="Public domain (LJ Speech dataset), trained from scratch.",
        dataset_url="https://keithito.com/LJ-Speech-Dataset/",
        model_card_url=(
            "https://huggingface.co/rhasspy/piper-voices/blob/main/"
            "en/en_US/ljspeech/medium/MODEL_CARD"
        ),
    ),
    "es_ES-davefx-medium": VoiceInfo(
        voice_id="es_ES-davefx-medium",
        language="es",
        language_name="Spanish",
        sample_rate=22_050,
        size_bytes=63_201_294,
        license_summary=(
            "Model card lists the dataset as CC0. The card also says this voice "
            "was finetuned from the English Lessac voice, whose dataset license "
            "is separate. Read the model card before redistributing the voice."
        ),
        dataset_url="https://github.com/OHF-Voice/voice-datasets",
        model_card_url=(
            "https://huggingface.co/rhasspy/piper-voices/blob/main/"
            "es/es_ES/davefx/medium/MODEL_CARD"
        ),
    ),
}

DEFAULT_VOICE_FOR_LANGUAGE = {
    "en": "en_US-ljspeech-medium",
    "es": "es_ES-davefx-medium",
}


class SpeechSynthesisError(RuntimeError):
    """A voice could not be downloaded, loaded, or played."""


def voice_for_language(language: str, voice_id: str | None = None) -> VoiceInfo:
    """Select a verified voice. An override must match the target language."""

    if voice_id is None:
        voice_id = DEFAULT_VOICE_FOR_LANGUAGE.get(language)
        if voice_id is None:
            known = ", ".join(sorted(DEFAULT_VOICE_FOR_LANGUAGE))
            raise SpeechSynthesisError(
                f"No verified Piper voice is configured for '{language}'. "
                f"Configured languages: {known}."
            )
    try:
        voice = VOICES[voice_id]
    except KeyError as exc:
        known = ", ".join(sorted(VOICES))
        raise SpeechSynthesisError(
            f"Unknown Piper voice '{voice_id}'. Verified voices: {known}."
        ) from exc
    if voice.language != language:
        raise SpeechSynthesisError(
            f"Voice '{voice.voice_id}' is {voice.language_name}, "
            f"not the requested target language '{language}'."
        )
    return voice


def voice_paths(voices_dir: Path, voice_id: str) -> tuple[Path, Path]:
    return voices_dir / f"{voice_id}.onnx", voices_dir / f"{voice_id}.onnx.json"


def voice_is_ready(voices_dir: Path, voice_id: str) -> bool:
    info = VOICES[voice_id]
    model_path, config_path = voice_paths(voices_dir, voice_id)
    if not model_path.is_file() or not config_path.is_file():
        return False
    return (
        model_path.stat().st_size >= int(info.size_bytes * 0.9)
        and config_path.stat().st_size > 100
    )


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, channels: int) -> None:
    """Write float audio using the synthesis sample rate and channel count."""

    samples = np.asarray(audio, dtype=np.float32)
    if channels == 1:
        samples = samples.reshape(-1)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class Speaker:
    """Cache one Piper voice and play it through a selected output device."""

    def __init__(
        self,
        voices_dir: Path | str = "models/tts",
        output_device: int | None = None,
        use_cuda: bool = False,
        voice_id: str | None = None,
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.output_device = output_device
        self.use_cuda = use_cuda
        self.voice_id_override = voice_id
        self._voice = None
        self._loaded_voice_id: str | None = None

    def download(self, language: str) -> VoiceInfo:
        """Download the voice with Piper's official helper if it is missing."""

        info = voice_for_language(language, self.voice_id_override)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        if voice_is_ready(self.voices_dir, info.voice_id):
            print(f"Piper voice already downloaded: {info.voice_id}")
            return info
        print(f"Downloading Piper voice {info.voice_id}...")

        self._remove_truncated(info)
        try:
            from piper.download_voices import download_voice
        except ImportError as exc:
            raise SpeechSynthesisError(
                "piper-tts is not installed. Install requirements before downloading voices."
            ) from exc
        try:
            download_voice(info.voice_id, self.voices_dir)
        except Exception as exc:
            self._remove_truncated(info)
            raise SpeechSynthesisError(
                "Could not download the Piper voice. Internet access is required "
                "for the first download. An incomplete file was removed; run the "
                f"download command again. Details: {exc}"
            ) from exc
        if not voice_is_ready(self.voices_dir, info.voice_id):
            self._remove_truncated(info)
            raise SpeechSynthesisError(
                f"The Piper voice '{info.voice_id}' did not finish downloading."
            )
        return info

    def load(self, language: str):
        """Load the ONNX voice once."""

        info = self.download(language)
        if self._voice is not None and self._loaded_voice_id == info.voice_id:
            return self._voice
        model_path, _config_path = voice_paths(self.voices_dir, info.voice_id)
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise SpeechSynthesisError(
                "piper-tts is not installed. Install requirements before speaking."
            ) from exc
        try:
            self._voice = PiperVoice.load(model_path, use_cuda=self.use_cuda)
        except Exception as exc:
            hint = ""
            if self.use_cuda:
                hint = " CUDA playback needs the onnxruntime-gpu package."
            raise SpeechSynthesisError(
                f"Could not load Piper voice '{info.voice_id}'.{hint} Details: {exc}"
            ) from exc
        self._loaded_voice_id = info.voice_id
        return self._voice

    def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int, int]:
        """Return float audio, its sample rate, and its channel count."""

        cleaned = " ".join(text.split())
        if not cleaned:
            raise SpeechSynthesisError("There is no text to speak.")
        voice = self.load(language)
        pieces: list[np.ndarray] = []
        sample_rate: int | None = None
        channels = 1
        try:
            for chunk in voice.synthesize(cleaned):
                sample_rate = int(chunk.sample_rate)
                channels = int(chunk.sample_channels)
                pieces.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
        except Exception as exc:
            raise SpeechSynthesisError(f"Piper could not synthesize speech: {exc}") from exc
        if not pieces or sample_rate is None:
            raise SpeechSynthesisError("Piper returned no audio.")
        audio = np.concatenate(pieces).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        return audio, sample_rate, channels

    def speak(self, text: str, language: str, save_path: Path | None = None) -> None:
        """Synthesize text and block until playback finishes."""

        audio, sample_rate, channels = self.synthesize(text, language)
        if save_path is not None:
            write_wav(save_path, audio, sample_rate, channels)
        self.play(audio, sample_rate)

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio at the sample rate stored in the generated waveform."""

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SpeechSynthesisError(
                "sounddevice is not installed, so audio cannot be played."
            ) from exc
        try:
            sd.stop()
            sd.play(audio, samplerate=sample_rate, device=self.output_device)
            sd.wait()
        except Exception as exc:
            raise SpeechSynthesisError(
                "Could not play audio on the selected output device. "
                "Run --list-devices and try another --output-device. "
                f"Details: {exc}"
            ) from exc

    def _remove_truncated(self, info: VoiceInfo) -> None:
        model_path, config_path = voice_paths(self.voices_dir, info.voice_id)
        minimum = int(info.size_bytes * 0.9)
        if model_path.exists() and model_path.stat().st_size < minimum:
            model_path.unlink()
        if config_path.exists() and config_path.stat().st_size < 50:
            config_path.unlink()
