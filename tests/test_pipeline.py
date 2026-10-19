# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that do not need a microphone, a speaker, or model weights."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asr
import nmt
import tts
import translator


class FakeRecognizer:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[str] = []

    def transcribe(self, audio, language: str) -> str:
        self.calls.append(language)
        if not self.texts:
            return ""
        return self.texts.pop(0)


class FakeTextTranslator:
    def __init__(self, results: dict[str, str] | None = None, fail_once: bool = False) -> None:
        self.results = results or {}
        self.fail_once = fail_once
        self.calls: list[str] = []
        self.pairs: list[tuple[str, str]] = []

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append(text)
        self.pairs.append((source_language, target_language))
        if self.fail_once:
            self.fail_once = False
            raise nmt.TranslationError("inference failed")
        return self.results.get(text, f"translated:{text}")


class FakeSpeaker:
    def __init__(self, recorder=None) -> None:
        self.recorder = recorder
        self.spoken: list[str] = []

    def speak(self, text: str, language: str, save_path=None) -> None:
        if self.recorder is not None:
            self.recorder.playback_active = True
        self.spoken.append(text)
        if self.recorder is not None:
            self.recorder.playback_active = False


class ScriptedRecorder:
    def __init__(self, audios: list) -> None:
        self.audios = list(audios)
        self.playback_active = False
        self.calls = 0

    def record_utterance(self):
        if self.playback_active:
            raise AssertionError("The microphone was read while speech was playing.")
        self.calls += 1
        if not self.audios:
            raise KeyboardInterrupt
        return self.audios.pop(0)


def loud_frame(samples: int, level: float = 0.2) -> np.ndarray:
    return np.full(samples, level, dtype=np.float32)


def quiet_frame(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)


class ModelSelectionTests(unittest.TestCase):
    def test_language_list_matches_the_model_card(self) -> None:
        self.assertEqual(len(nmt.M2M100_LANGUAGE_CODES), 100)
        for code, _name in nmt.TUTORIAL_LANGUAGES:
            self.assertIn(code, nmt.M2M100_LANGUAGE_CODES)

    def test_every_direction_uses_one_model(self) -> None:
        self.assertEqual(nmt.model_name_for("en", "es"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("es", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("fr", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("vi", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("ja", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("de", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("ko", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("zh", "en"), "facebook/m2m100_418M")
        self.assertEqual(nmt.model_name_for("en", "fr"), "facebook/m2m100_418M")

    def test_target_voices_match_direction(self) -> None:
        self.assertEqual(tts.voice_for_language("es").voice_id, "es_ES-davefx-medium")
        self.assertEqual(tts.voice_for_language("en").voice_id, "en_US-ljspeech-medium")
        self.assertEqual(tts.voice_for_language("fr").voice_id, "fr_FR-siwis-medium")
        self.assertEqual(tts.voice_for_language("vi").voice_id, "vi_VN-vais1000-medium")
        self.assertEqual(tts.voice_for_language("de").voice_id, "de_DE-thorsten-medium")
        self.assertEqual(tts.voice_for_language("ko").voice_id, "ko_KR-kss-medium")
        self.assertFalse(tts.voice_available("ja"))
        self.assertFalse(tts.voice_available("zh"))

    def test_unsupported_language_pair(self) -> None:
        with self.assertRaises(nmt.UnsupportedLanguagePair):
            nmt.model_name_for("xx", "en")

    def test_same_language_is_not_offered(self) -> None:
        with self.assertRaises(nmt.UnsupportedLanguagePair):
            nmt.model_name_for("en", "en")

    def test_unknown_voice_is_rejected(self) -> None:
        with self.assertRaises(tts.SpeechSynthesisError):
            tts.voice_for_language("es", "en_US-ljspeech-medium")

    def test_chinese_speech_asks_for_text_only(self) -> None:
        with self.assertRaises(tts.SpeechSynthesisError) as caught:
            tts.voice_for_language("zh")
        self.assertIn("--no-tts", str(caught.exception))


class TranslationBehaviorTests(unittest.TestCase):
    def test_empty_text_does_not_load_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            translator_model = nmt.TextTranslator(directory)
            self.assertEqual(translator_model.translate("   ", "en", "es"), "")
            self.assertIsNone(translator_model._model)

    def test_translation_failure_is_reported(self) -> None:
        torch = self._torch()
        translator_model = nmt.TextTranslator("models/nmt")

        class BrokenModel:
            def generate(self, **kwargs):
                raise RuntimeError("inference failed")

        class Tokenizer:
            src_lang = None

            def get_lang_id(self, lang):
                return 7

            def __call__(self, text, return_tensors, truncation, max_length):
                return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

            def decode(self, tokens, skip_special_tokens=True):
                return "should not be used"

        translator_model._tokenizer = Tokenizer()
        translator_model._model = BrokenModel()
        with self.assertRaises(nmt.TranslationError):
            translator_model.translate("hello", "en", "es")

    def test_translation_uses_the_decoded_text(self) -> None:
        torch = self._torch()
        translator_model = nmt.TextTranslator("models/nmt")

        class Model:
            def __init__(self) -> None:
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.ones((1, 3), dtype=torch.long)

        class Tokenizer:
            def __init__(self) -> None:
                self.src_lang = None

            def get_lang_id(self, lang):
                return {"en": 4, "es": 7, "fr": 8, "ja": 9, "vi": 10}[lang]

            def __call__(self, text, return_tensors, truncation, max_length):
                self.encoded_source = self.src_lang
                return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

            def decode(self, tokens, skip_special_tokens=True):
                return "  Hola, ¿cómo estás?  "

        model = Model()
        tokenizer = Tokenizer()
        translator_model._tokenizer = tokenizer
        translator_model._model = model
        self.assertEqual(
            translator_model.translate("Hello, how are you?", "en", "es"),
            "Hola, ¿cómo estás?",
        )
        self.assertEqual(tokenizer.encoded_source, "en")
        self.assertEqual(model.kwargs["forced_bos_token_id"], 7)

    def test_japanese_and_french_set_their_own_language_tokens(self) -> None:
        torch = self._torch()
        translator_model = nmt.TextTranslator("models/nmt")
        seen = []

        class Model:
            def generate(self, **kwargs):
                seen.append(kwargs["forced_bos_token_id"])
                return torch.ones((1, 2), dtype=torch.long)

        class Tokenizer:
            def __init__(self) -> None:
                self.src_lang = None

            def get_lang_id(self, lang):
                return {"en": 4, "fr": 8, "ja": 9, "vi": 10}[lang]

            def __call__(self, text, return_tensors, truncation, max_length):
                seen.append(self.src_lang)
                return {"input_ids": torch.ones((1, 2), dtype=torch.long)}

            def decode(self, tokens, skip_special_tokens=True):
                return "hello"

        translator_model._tokenizer = Tokenizer()
        translator_model._model = Model()
        translator_model.translate("こんにちは", "ja", "en")
        translator_model.translate("Bonjour", "fr", "en")
        translator_model.translate("Xin chào", "vi", "en")
        self.assertEqual(seen, ["ja", 4, "fr", 4, "vi", 4])

    def test_model_object_is_loaded_once(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers is not installed")
        with tempfile.TemporaryDirectory() as directory:
            translator_model = nmt.TextTranslator(directory)
            with patch("nmt.load_pretrained") as load_pretrained:
                model = unittest.mock.Mock()
                load_pretrained.return_value = (object(), model)
                translator_model.load("en", "es")
                translator_model.load("ja", "en")
                translator_model.load("fr", "en")
                self.assertEqual(load_pretrained.call_count, 1)

    def _torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        return torch


class SpeechDetectionTests(unittest.TestCase):
    def make_detector(self, **overrides) -> asr.EnergyUtteranceDetector:
        settings = dict(
            sample_rate=16_000,
            threshold=0.02,
            silence_seconds=0.3,
            min_speech_seconds=0.4,
            max_speech_seconds=2.0,
            pre_roll_seconds=0.0,
        )
        settings.update(overrides)
        return asr.EnergyUtteranceDetector(asr.DetectorConfig(**settings))

    def test_silence_is_ignored(self) -> None:
        detector = self.make_detector()
        for _ in range(5):
            self.assertEqual(detector.process_frame(quiet_frame(1600)), "listening")
        self.assertIsNone(detector.take_audio())

    def test_short_noise_is_discarded(self) -> None:
        detector = self.make_detector()
        self.assertEqual(detector.process_frame(loud_frame(1600)), "recording")
        statuses = [detector.process_frame(quiet_frame(1600)) for _ in range(3)]
        self.assertEqual(statuses[-1], "discarded")
        self.assertIsNone(detector.take_audio())

    def test_a_full_phrase_is_returned(self) -> None:
        detector = self.make_detector()
        for _ in range(5):
            self.assertEqual(detector.process_frame(loud_frame(1600)), "recording")
        for _ in range(2):
            self.assertEqual(detector.process_frame(quiet_frame(1600)), "recording")
        self.assertEqual(detector.process_frame(quiet_frame(1600)), "finished")
        audio = detector.take_audio()
        self.assertIsNotNone(audio)
        assert audio is not None
        self.assertEqual(audio.size, 1600 * 8)
        self.assertGreater(asr.rms(audio), 0.02)

    def test_maximum_duration_stops_the_recording(self) -> None:
        detector = self.make_detector(max_speech_seconds=0.5, min_speech_seconds=0.2)
        status = "listening"
        for _ in range(5):
            status = detector.process_frame(loud_frame(1600))
        self.assertEqual(status, "finished")
        audio = detector.take_audio()
        assert audio is not None
        self.assertEqual(audio.size, 1600 * 5)

    def test_detector_can_capture_a_second_phrase(self) -> None:
        detector = self.make_detector(min_speech_seconds=0.1, silence_seconds=0.1)
        self.assertEqual(detector.process_frame(loud_frame(1600)), "recording")
        self.assertEqual(detector.process_frame(quiet_frame(1600)), "finished")
        first = detector.take_audio()
        self.assertEqual(detector.process_frame(loud_frame(1600, 0.3)), "recording")
        self.assertEqual(detector.process_frame(quiet_frame(1600)), "finished")
        second = detector.take_audio()
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)

    def test_short_or_low_confidence_detection_is_rejected(self) -> None:
        recognizer = asr.SpeechRecognizer()
        short_clip = np.ones(16_000, dtype=np.float32)
        long_clip = np.ones(16_000 * 2, dtype=np.float32)
        with self.assertRaises(asr.UncertainLanguageError):
            recognizer._require_confident_detection(short_clip, "ja", 0.70)
        with self.assertRaises(asr.UncertainLanguageError):
            recognizer._require_confident_detection(long_clip, "fr", 0.40)
        recognizer._require_confident_detection(long_clip, "vi", 0.70)
        recognizer._require_confident_detection(short_clip, "en", 0.85)

    def test_unknown_speech_language_is_rejected_before_loading(self) -> None:
        recognizer = asr.SpeechRecognizer()
        with self.assertRaises(ValueError):
            recognizer.transcribe(None, "xx")
        self.assertIsNone(recognizer._model)

    def test_empty_and_silent_audio_are_not_transcribed(self) -> None:
        recognizer = asr.SpeechRecognizer()
        self.assertEqual(recognizer.transcribe(None, "en"), "")
        self.assertEqual(recognizer.transcribe(np.zeros(16_000, dtype=np.float32), "es"), "")
        self.assertIsNone(recognizer._model)
        self.assertEqual(asr.clean_transcript(" ... "), "")

    def test_resample_changes_the_length(self) -> None:
        audio = np.zeros(4800, dtype=np.float32)
        resampled = asr.resample_audio(audio, 48_000, 16_000)
        self.assertEqual(resampled.size, 1600)

    def test_asr_model_loads_once(self) -> None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            self.skipTest("faster-whisper is not installed")
        with tempfile.TemporaryDirectory() as directory:
            recognizer = asr.SpeechRecognizer("tiny", directory)
            with patch("faster_whisper.WhisperModel") as whisper_cls:
                recognizer.load()
                recognizer.load()
                self.assertEqual(whisper_cls.call_count, 1)
                self.assertEqual(whisper_cls.call_args.kwargs["device"], "cpu")
                self.assertEqual(whisper_cls.call_args.kwargs["compute_type"], "int8")


class DeviceAndCacheTests(unittest.TestCase):
    def test_device_table_lists_input_and_output(self) -> None:
        devices = [
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        ]
        table = translator.format_device_table(devices)
        rows = [line.split() for line in table.splitlines() if line[:1].isdigit()]
        self.assertEqual(rows[0], ["0", "1", "0", "Microphone"])
        self.assertEqual(rows[1], ["1", "0", "2", "Speakers"])

    def test_invalid_device_index_is_rejected(self) -> None:
        fake_devices = [
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0},
        ]
        with patch("sounddevice.query_devices", return_value=fake_devices):
            with self.assertRaises(SystemExit):
                translator.validate_audio_device(3, "input")
            with self.assertRaises(SystemExit):
                translator.validate_audio_device(0, "output")

    def test_model_directories_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = translator.prepare_model_directories(Path(directory) / "models")
            self.assertTrue(paths["asr"].is_dir())
            self.assertTrue(paths["nmt"].is_dir())
            self.assertTrue(paths["tts"].is_dir())

    def test_truncated_voice_is_not_treated_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, config = tts.voice_paths(root, "es_ES-davefx-medium")
            model.write_bytes(b"partial")
            config.write_text("{}", encoding="utf-8")
            self.assertFalse(tts.voice_is_ready(root, "es_ES-davefx-medium"))

    def test_failed_voice_download_removes_a_partial_file(self) -> None:
        try:
            import piper.download_voices  # noqa: F401
        except ImportError:
            self.skipTest("piper-tts is not installed")

        def fail_download(voice: str, download_dir: Path, force_redownload: bool = False) -> None:
            (Path(download_dir) / f"{voice}.onnx").write_bytes(b"x" * 100)
            raise ConnectionError("offline")

        with tempfile.TemporaryDirectory() as directory:
            speaker = tts.Speaker(voices_dir=directory)
            with patch("piper.download_voices.download_voice", fail_download):
                with self.assertRaises(tts.SpeechSynthesisError):
                    speaker.download("es")
            self.assertFalse((Path(directory) / "es_ES-davefx-medium.onnx").exists())

    def test_voice_loads_once_when_the_file_is_present(self) -> None:
        try:
            import piper  # noqa: F401
        except ImportError:
            self.skipTest("piper-tts is not installed")
        with tempfile.TemporaryDirectory() as directory:
            speaker = tts.Speaker(voices_dir=directory)
            with patch("tts.voice_is_ready", return_value=True), patch("piper.PiperVoice") as voice_cls:
                speaker.load("en")
                speaker.load("en")
                self.assertEqual(voice_cls.load.call_count, 1)


class CommandLineTests(unittest.TestCase):
    def test_documented_commands_parse(self) -> None:
        parser = translator.build_parser()
        parser.parse_args(["--source", "en", "--target", "es"])
        parser.parse_args(["--source", "es", "--target", "en"])
        parser.parse_args(["--list-devices"])
        parser.parse_args(["--download-models", "--source", "en", "--target", "es"])
        parser.parse_args(["--source", "en", "--target", "es", "--input-device", "1", "--output-device", "3"])
        text_args = parser.parse_args(["--source", "en", "--target", "es", "--text", "Hello, how are you?"])
        self.assertEqual(text_args.text, "Hello, how are you?")
        parser.parse_args(["--source", "en", "--target", "es", "--asr-model", "small"])
        parser.parse_args(["--source", "en", "--target", "es", "--no-tts"])
        parser.parse_args(["--source", "ja", "--target", "en"])
        parser.parse_args(["--source", "fr", "--target", "en"])
        parser.parse_args(["--source", "vi", "--target", "en"])
        parser.parse_args(["--source", "en", "--target", "fr"])
        parser.parse_args(["--source", "auto", "--target", "en"])
        parser.parse_args(["--list-languages"])
        parser.parse_args(["--source", "ja", "--target", "en", "--text", "こんにちは", "--no-tts"])
        with self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_invalid_language_is_rejected(self) -> None:
        code = translator.main(["--source", "xx", "--target", "en", "--no-tts", "--text", "hello"])
        self.assertEqual(code, 2)

    def test_chinese_speech_without_text_only_flag_exits(self) -> None:
        code = translator.main(["--source", "en", "--target", "zh", "--text", "hello"])
        self.assertEqual(code, 2)
        code = translator.main(["--source", "en", "--target", "ja", "--text", "hello"])
        self.assertEqual(code, 2)

    def test_automatic_detection_rejects_typed_text(self) -> None:
        code = translator.main(["--source", "auto", "--target", "en", "--no-tts", "--text", "hello"])
        self.assertEqual(code, 2)

    def test_invalid_timing_is_rejected(self) -> None:
        parser = translator.build_parser()
        args = parser.parse_args(["--min-speech-duration", "10", "--max-speech-duration", "2"])
        with self.assertRaises(SystemExit):
            translator.validate_runtime_arguments(args)

    def test_same_language_command_exits(self) -> None:
        code = translator.main(["--source", "en", "--target", "en", "--no-tts", "--text", "hello"])
        self.assertEqual(code, 2)

    def test_readme_contains_the_documented_commands(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for command in (
            ".\\.venv\\Scripts\\python.exe translator.py --source en --target es",
            ".\\.venv\\Scripts\\python.exe translator.py --source es --target en",
            ".\\.venv\\Scripts\\python.exe translator.py --list-devices",
            ".\\.venv\\Scripts\\python.exe translator.py --download-models --source en --target es",
            ".\\.venv\\Scripts\\python.exe translator.py --source en --target es --input-device 1 --output-device 3",
            '.\\.venv\\Scripts\\python.exe translator.py --source en --target es --text "Hello, how are you?"',
            ".\\.venv\\Scripts\\python.exe translator.py --source en --target es --asr-model small",
            ".\\.venv\\Scripts\\python.exe translator.py --help",
            ".\\.venv\\Scripts\\python.exe translator.py --source en --target es --text \"Hello, how are you?\" --no-tts",
            ".\\.venv\\Scripts\\python.exe translator.py --source ja --target en",
            ".\\.venv\\Scripts\\python.exe translator.py --source fr --target en",
            ".\\.venv\\Scripts\\python.exe translator.py --source vi --target en",
            ".\\.venv\\Scripts\\python.exe translator.py --source en --target fr",
            ".\\.venv\\Scripts\\python.exe translator.py --source ja --target en --text \"こんにちは\" --no-tts",
            ".\\.venv\\Scripts\\python.exe translator.py --list-languages",
            ".\\.venv\\Scripts\\python.exe translator.py --source auto --target en",
        ):
            self.assertIn(command, readme)

    def test_requirements_match_the_stack(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for package in ("numpy", "sounddevice", "faster-whisper", "transformers", "sentencepiece", "piper-tts"):
            self.assertIn(package, requirements)

    def test_gitignore_excludes_models_and_outputs(self) -> None:
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("models/*", ignored)
        self.assertIn("outputs/*", ignored)
        self.assertIn(".venv/", ignored)


class PipelineCoordinationTests(unittest.TestCase):
    def test_two_phrases_are_translated_and_then_the_loop_stops(self) -> None:
        recorder = ScriptedRecorder([np.zeros(10), np.zeros(10)])
        recognizer = FakeRecognizer(["hello", "goodbye"])
        text_translator = FakeTextTranslator({"hello": "hola", "goodbye": "adios"})
        speaker = FakeSpeaker(recorder)
        session = translator.PhraseTranslator(recognizer, text_translator, speaker, "en", "es")
        translator.listen_forever(recorder, session, playback_gap=0)
        self.assertEqual(speaker.spoken, ["hola", "adios"])
        self.assertEqual(text_translator.calls, ["hello", "goodbye"])
        self.assertEqual(recorder.calls, 3)

    def test_empty_recognition_does_not_translate_or_speak(self) -> None:
        recorder = ScriptedRecorder([np.zeros(10)])
        recognizer = FakeRecognizer([""])
        text_translator = FakeTextTranslator()
        speaker = FakeSpeaker(recorder)
        session = translator.PhraseTranslator(recognizer, text_translator, speaker, "en", "es")
        translator.listen_forever(recorder, session, playback_gap=0)
        self.assertEqual(text_translator.calls, [])
        self.assertEqual(speaker.spoken, [])

    def test_a_translation_error_does_not_stop_the_next_phrase(self) -> None:
        recorder = ScriptedRecorder([np.zeros(10), np.zeros(10)])
        recognizer = FakeRecognizer(["hello", "goodbye"])
        text_translator = FakeTextTranslator({"goodbye": "adios"}, fail_once=True)
        speaker = FakeSpeaker()
        session = translator.PhraseTranslator(recognizer, text_translator, speaker, "en", "es")
        translator.listen_forever(recorder, session, playback_gap=0)
        self.assertEqual(speaker.spoken, ["adios"])

    def test_no_tts_does_not_speak(self) -> None:
        recognizer = FakeRecognizer(["hello"])
        text_translator = FakeTextTranslator({"hello": "hola"})
        speaker = FakeSpeaker()
        session = translator.PhraseTranslator(
            recognizer,
            text_translator,
            speaker,
            "en",
            "es",
            use_tts=False,
        )
        original, translated = session.process(np.zeros(10))
        self.assertEqual((original, translated), ("hello", "hola"))
        self.assertEqual(speaker.spoken, [])

    def test_automatic_detection_uses_the_detected_source(self) -> None:
        class DetectingRecognizer:
            def transcribe_with_language(self, audio, language):
                self.language = language
                return "bonjour", "fr", 0.91

        recognizer = DetectingRecognizer()
        text_translator = FakeTextTranslator({"bonjour": "hello"})
        speaker = FakeSpeaker()
        session = translator.PhraseTranslator(recognizer, text_translator, speaker, "auto", "en")
        original, translated = session.process(np.zeros(10))
        self.assertEqual((original, translated), ("bonjour", "hello"))
        self.assertEqual(recognizer.language, "auto")
        self.assertEqual(text_translator.pairs, [("fr", "en")])
        self.assertEqual(speaker.spoken, ["hello"])

    def test_uncertain_detection_does_not_translate(self) -> None:
        class UnsureRecognizer:
            def transcribe_with_language(self, audio, language):
                raise asr.UncertainLanguageError("not sure")

        recorder = ScriptedRecorder([np.zeros(10)])
        text_translator = FakeTextTranslator()
        speaker = FakeSpeaker()
        session = translator.PhraseTranslator(UnsureRecognizer(), text_translator, speaker, "auto", "en")
        translator.listen_forever(recorder, session, playback_gap=0)
        self.assertEqual(text_translator.calls, [])
        self.assertEqual(speaker.spoken, [])

    def test_detected_target_language_is_not_translated(self) -> None:
        class EnglishRecognizer:
            def transcribe_with_language(self, audio, language):
                return "hello", "en", 0.95

        text_translator = FakeTextTranslator()
        speaker = FakeSpeaker()
        session = translator.PhraseTranslator(EnglishRecognizer(), text_translator, speaker, "auto", "en")
        with self.assertRaises(nmt.UnsupportedLanguagePair):
            session.process(np.ones(10))
        self.assertEqual(text_translator.calls, [])
        self.assertEqual(speaker.spoken, [])


if __name__ == "__main__":
    unittest.main()
