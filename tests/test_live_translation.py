# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-model checks. Skipped unless NOMINALLY_RUN_LIVE_TESTS=1.

These tests download or reuse the M2M100 weights and Piper voices. They do
not open the microphone.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

LIVE = os.environ.get("NOMINALLY_RUN_LIVE_TESTS") == "1"


def working_set_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.K32GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error())
    return counters.WorkingSetSize / (1024 * 1024)


@unittest.skipUnless(LIVE, "Set NOMINALLY_RUN_LIVE_TESTS=1 to run real model checks.")
class LiveTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import nmt

        cls.translator = nmt.TextTranslator(ROOT / "models" / "nmt", "cpu")
        started = time.perf_counter()
        cls.translator.load("en", "es")
        cls.load_seconds = time.perf_counter() - started
        cls.memory_after_load = working_set_mb()
        print(f"LOAD_SECONDS {cls.load_seconds:.2f}")
        print(f"WORKING_SET_MB {cls.memory_after_load:.0f}")

    def test_weights_are_not_duplicated(self) -> None:
        root = ROOT / "models" / "nmt" / "models--facebook--m2m100_418M"
        large = []
        for path in root.rglob("*"):
            if not path.is_file() or path.stat().st_size < 100_000_000:
                continue
            large.append((path.name, path.stat().st_size))
        names = [name for name, _size in large]
        self.assertIn("pytorch_model.bin", names)
        self.assertNotIn("model.safetensors", names)
        self.assertNotIn("rust_model.ot", names)
        bins = [size for name, size in large if name == "pytorch_model.bin"]
        self.assertEqual(len(bins), 1)
        self.assertGreater(bins[0], 1_900_000_000)
        print("NMT_BIN_BYTES", bins[0])

    def test_tutorial_sentences(self) -> None:
        samples = [
            ("en", "es", "Hello, how are you?", ("hola",)),
            ("es", "en", "Hola, ¿cómo estás?", ("how", "hello")),
            ("fr", "en", "Bonjour, comment allez-vous ?", ("hello", "how", "good")),
            ("vi", "en", "Xin chào, bạn khỏe không?", ("hello", "how", "hi")),
            ("ja", "en", "こんにちは、お元気ですか？", ("hello", "how", "hi")),
            ("de", "en", "Guten Tag, wie geht es Ihnen?", ("hello", "how", "good", "day")),
            ("ko", "en", "안녕하세요, 잘 지내세요?", ("hello", "how", "hi")),
            ("zh", "en", "你好，你好吗？", ("hello", "how", "hi")),
            ("en", "fr", "Hello, how are you?", ("bonjour", "comment", "salut")),
            ("en", "vi", "Hello, how are you?", ("chào", "khỏe", "xin")),
            ("en", "ja", "Hello, how are you?", ("こんにちは", "元気", "ですか")),
            ("en", "zh", "Hello, how are you?", ("您好", "你好", "怎么样")),
        ]
        for source, target, text, needles in samples:
            started = time.perf_counter()
            translated = self.translator.translate(text, source, target)
            elapsed = time.perf_counter() - started
            print(f"TRANSLATION {source}->{target} {elapsed:.2f}s | {translated}")
            folded = translated.casefold()
            self.assertTrue(translated.strip(), f"empty translation for {source}->{target}")
            self.assertTrue(
                any(needle.casefold() in folded for needle in needles),
                f"{source}->{target} produced {translated!r}",
            )
        print(f"WORKING_SET_AFTER_TRANSLATION_MB {working_set_mb():.0f}")

    def test_repeated_calls_do_not_reload(self) -> None:
        model = self.translator._model
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            translated = self.translator.translate("Good morning.", "en", "es")
            timings.append(time.perf_counter() - started)
            self.assertIn("buen", translated.casefold())
        self.assertIs(self.translator._model, model)
        print("WARM_EN_ES_SECONDS", " ".join(f"{item:.2f}" for item in timings))

    def test_empty_text_stays_empty(self) -> None:
        self.assertEqual(self.translator.translate("   ", "ja", "en"), "")


@unittest.skipUnless(LIVE, "Set NOMINALLY_RUN_LIVE_TESTS=1 to run real model checks.")
class LiveSpeechTests(unittest.TestCase):
    def test_synthesized_speech_round_trip(self) -> None:
        import asr
        import nmt
        import tts

        speaker = tts.Speaker(ROOT / "models" / "tts")
        recognizer = asr.SpeechRecognizer("base", ROOT / "models" / "asr", "cpu")
        translator = nmt.TextTranslator(ROOT / "models" / "nmt", "cpu")
        translator.load()
        prompts = [
            ("fr", "Bonjour. Comment allez-vous?"),
            ("vi", "Xin chào. Bạn khỏe không?"),
            ("es", "Hola. ¿Cómo estás?"),
            ("de", "Guten Tag. Wie geht es Ihnen?"),
            ("ko", "안녕하세요. 잘 지내세요?"),
        ]
        for language, sentence in prompts:
            started = time.perf_counter()
            audio, sample_rate, channels = speaker.synthesize(sentence, language)
            tts_seconds = time.perf_counter() - started
            self.assertEqual(channels, 1)
            self.assertGreater(sample_rate, 8_000)
            self.assertGreater(float(abs(audio).max()), 0.01)
            samples = asr.resample_audio(audio.reshape(-1), sample_rate, 16_000)
            started = time.perf_counter()
            heard = recognizer.transcribe(samples, language)
            asr_seconds = time.perf_counter() - started
            started = time.perf_counter()
            english = translator.translate(heard, language, "en")
            nmt_seconds = time.perf_counter() - started
            print(
                f"ROUND {language} tts {tts_seconds:.2f}s asr {asr_seconds:.2f}s "
                f"nmt {nmt_seconds:.2f}s | heard {heard!r} | en {english!r}"
            )
            self.assertTrue(heard.strip(), f"no transcript for {language}")
            self.assertTrue(english.strip(), f"no English translation for {language}")

        english_audio, english_rate, _channels = speaker.synthesize("Hello. How are you?", "en")
        import sounddevice as sd

        sd.stop()
        sd.play(english_audio, samplerate=english_rate)
        sd.wait()
        print("PLAYBACK_OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
