# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phrase-by-phrase speech translator.

The microphone stays closed while translated speech is playing. Listening
resumes after playback, so the program can handle the next phrase without
being restarted. This is sequential translation, not full-duplex interpretation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Windows without Developer Mode cannot symlink the Hugging Face cache.
# Downloads still work. HF_HUB_DISABLE_XET avoids an optional-package warning;
# the models still download over HTTPS. Set both before the hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

PROJECT_ROOT = Path(__file__).resolve().parent


def configure_console() -> None:
    """Let Spanish punctuation print on Windows consoles that default to cp1252."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


def prepare_model_directories(root: Path | None = None) -> dict[str, Path]:
    base = Path(root) if root is not None else PROJECT_ROOT / "models"
    paths = {
        "root": base,
        "asr": base / "asr",
        "nmt": base / "nmt",
        "tts": base / "tts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def missing_dependency_message(exc: ImportError) -> str:
    return (
        "A required library is not installed.\n"
        "Create a virtual environment, then install the CPU version of PyTorch "
        "and the project requirements. The program does not install packages itself.\n"
        "  py -3.12 -m venv .venv\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install --upgrade \"pip>=23.1,<26\"\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        f"Import error: {exc}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Listen to a phrase, translate it, and speak the result locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python translator.py --source en --target es\n"
            "  python translator.py --source es --target en\n"
            "  python translator.py --source ja --target en\n"
            "  python translator.py --source fr --target en\n"
            "  python translator.py --source vi --target en\n"
            "  python translator.py --source en --target fr\n"
            "  python translator.py --source auto --target en\n"
            "  python translator.py --list-languages\n"
            "  python translator.py --list-devices\n"
            "  python translator.py --download-models --source en --target es\n"
            "  python translator.py --source en --target es --input-device 1 --output-device 3\n"
            '  python translator.py --source en --target es --text "Hello, how are you?"\n'
            '  python translator.py --source ja --target en --text "こんにちは" --no-tts\n'
            "  python translator.py --source en --target es --asr-model small\n"
            "  python translator.py --source en --target es --no-tts\n"
        ),
    )
    parser.add_argument(
        "--source",
        default="en",
        help="Language you will speak. Use a code from --list-languages, or auto.",
    )
    parser.add_argument(
        "--target",
        default="es",
        help="Language you want to hear. Use a code from --list-languages.",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print microphones and speakers, then exit.")
    parser.add_argument("--list-languages", action="store_true", help="Print languages and which ones can be spoken, then exit.")
    parser.add_argument("--download-models", action="store_true", help="Download the ASR, translation, and TTS models, then exit.")
    parser.add_argument("--input-device", type=int, default=None, help="Microphone index from --list-devices.")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker index from --list-devices.")
    parser.add_argument("--text", default=None, help="Translate typed text instead of using the microphone.")
    parser.add_argument("--asr-model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Faster-Whisper model size. Default: base.")
    parser.add_argument("--no-tts", action="store_true", help="Print the translation without generating speech.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Run models on CPU (default) or CUDA.")
    parser.add_argument("--speech-threshold", type=float, default=None, help="Fixed microphone energy threshold. Skips ambient calibration.")
    parser.add_argument("--silence-duration", type=float, default=0.8, help="Seconds of quiet that end a phrase.")
    parser.add_argument("--min-speech-duration", type=float, default=0.4, help="Ignore sounds shorter than this many seconds.")
    parser.add_argument("--max-speech-duration", type=float, default=20.0, help="Maximum seconds recorded for one phrase.")
    parser.add_argument("--playback-gap", type=float, default=0.4, help="Seconds to wait after speech playback before listening again.")
    parser.add_argument("--tts-voice", default=None, help="Optional verified Piper voice id for the target language.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory for downloaded models. Default: ./models")
    parser.add_argument("--save-audio", action="store_true", help="Also write spoken audio to the outputs directory.")
    parser.add_argument("--verbose", action="store_true", help="Print short notes when a noise burst is ignored.")
    return parser


def validate_runtime_arguments(args: argparse.Namespace) -> None:
    if args.silence_duration <= 0 or args.min_speech_duration <= 0 or args.max_speech_duration <= 0:
        raise SystemExit("Speech timing values must be greater than zero.")
    if args.min_speech_duration >= args.max_speech_duration:
        raise SystemExit("--min-speech-duration must be smaller than --max-speech-duration.")
    if args.playback_gap < 0:
        raise SystemExit("--playback-gap cannot be negative.")
    if args.speech_threshold is not None and args.speech_threshold <= 0:
        raise SystemExit("--speech-threshold must be greater than zero.")


def list_audio_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(missing_dependency_message(exc)) from exc
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise SystemExit(
            "PortAudio could not list audio devices. On Windows, allow microphone "
            "access in Privacy settings and reinstall sounddevice. On Linux you may "
            f"also need the PortAudio system library. Details: {exc}"
        ) from exc
    print(format_device_table(devices))
    print()
    print("Use the index in the first column with --input-device or --output-device.")
    print("An input device needs a non-zero Input count. An output device needs a non-zero Output count.")


def display_device_name(name: str) -> str:
    """Keep one device on one line. Windows Bluetooth names can contain breaks."""

    return " ".join(str(name).split())


def format_device_table(devices) -> str:
    lines = ["Index  Input  Output  Name", "-----  -----  ------  ----"]
    for index, device in enumerate(devices):
        lines.append(
            f"{index:<5}  {int(device['max_input_channels']):<5}  "
            f"{int(device['max_output_channels']):<6}  {display_device_name(device['name'])}"
        )
    if len(lines) == 2:
        lines.append("(No audio devices were reported.)")
    return "\n".join(lines)


def validate_audio_device(device_index: int | None, kind: str) -> None:
    if device_index is None:
        return
    import sounddevice as sd

    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        raise SystemExit(f"Could not query audio devices: {exc}") from exc
    if device_index < 0 or device_index >= len(devices):
        raise SystemExit(
            f"{kind.capitalize()} device {device_index} does not exist. Run --list-devices."
        )
    device = devices[device_index]
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    if int(device[channel_key]) < 1:
        raise SystemExit(
            f"Device {device_index} ({device['name']}) has no {kind} channels. "
            "Run --list-devices and choose a different index."
        )


def device_label(device_index: int | None, kind: str) -> str:
    if device_index is None:
        return "Default Microphone" if kind == "input" else "Default Speakers"
    import sounddevice as sd

    info = sd.query_devices(device_index)
    return display_device_name(info["name"])


def language_label(code: str) -> str:
    if code == "auto":
        return "Automatic detection"
    from nmt import language_name

    return language_name(code)


def print_banner(source: str, target: str, input_label: str, output_label: str) -> None:
    print("===================================")
    print(" NOMINALLY - AUTO TRANSLATION")
    print("===================================")
    print()
    print(f"Source Language: {language_label(source)}")
    print(f"Target Language: {language_label(target)}")
    print(f"Input Device: {input_label}")
    print(f"Output Device: {output_label}")
    print()


def list_supported_languages() -> None:
    from nmt import LANGUAGE_NAMES, M2M100_LANGUAGE_CODES, MODEL_ID, TUTORIAL_LANGUAGES
    from tts import voice_available

    print("One translation model covers every language pair:")
    print(f"  {MODEL_ID}")
    print("Changing --source or --target does not download another translation model.")
    print()
    print("Code  Language      Speech recognition  Spoken reply")
    print("----  --------      ------------------  ------------")
    for code, name in TUTORIAL_LANGUAGES:
        spoken = "yes" if voice_available(code) else "no, use --no-tts"
        print(f"{code:<4}  {name:<12}  yes                 {spoken}")
    print()
    print("Spoken reply means a checked Piper voice exists for that target language.")
    print("Japanese and Chinese can be translated. This project does not speak them.")
    print("Japanese speech needs pyopenjtalk, which has no Windows install file.")
    print("The published Chinese Piper card does not state a license.")
    print("The Korean voice is non-commercial. See Credits in the README.")
    print()
    others = sorted(M2M100_LANGUAGE_CODES.difference(LANGUAGE_NAMES))
    print("Other text-translation codes (use --no-tts unless a voice is listed above):")
    print(", ".join(others))
    print()
    print("Automatic source detection: --source auto --target en")
    print("Explicit source selection is more predictable for a demonstration.")


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(missing_dependency_message(exc)) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but this PyTorch install cannot see a GPU. "
            "The default CPU install does not include CUDA. See the GPU section of the README, "
            "or run without --device cuda."
        )
    return "cuda"


def output_file_for(text: str) -> Path:
    safe = "".join(character if character.isalnum() else "_" for character in text[:40]).strip("_")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{safe or 'translation'}.wav"
    return PROJECT_ROOT / "outputs" / name


class PhraseTranslator:
    """Run recognition, translation, and optional speech for one utterance."""

    def __init__(
        self,
        recognizer,
        translator,
        speaker,
        source_language: str,
        target_language: str,
        use_tts: bool = True,
    ) -> None:
        self.recognizer = recognizer
        self.translator = translator
        self.speaker = speaker
        self.source_language = source_language
        self.target_language = target_language
        self.use_tts = use_tts

    def process(self, audio, before_speech=None, save_path: Path | None = None) -> tuple[str, str]:
        """Transcribe, translate, then speak.

        ``before_speech`` runs after the text exists and before playback, so the
        terminal can show the translation first. It may return a WAV path.
        """

        from nmt import M2M100_LANGUAGE_CODES, UnsupportedLanguagePair, language_name

        detected_probability = None
        if self.source_language == "auto":
            original, source_language, detected_probability = self.recognizer.transcribe_with_language(
                audio,
                "auto",
            )
        else:
            original = self.recognizer.transcribe(audio, self.source_language)
            source_language = self.source_language
        if not original or source_language is None:
            return "", ""
        if source_language not in M2M100_LANGUAGE_CODES:
            raise UnsupportedLanguagePair(
                f"Detected '{source_language}', and the translation model does not "
                "support that language. Set --source to the language you are speaking."
            )
        if source_language == self.target_language:
            raise UnsupportedLanguagePair(
                f"The spoken language is already {language_name(source_language)}. "
                "Nothing was translated. Choose a different --target, or speak another language."
            )
        if detected_probability is not None:
            print(
                f"Detected language: {language_name(source_language)} "
                f"({detected_probability:.0%})"
            )
        translated = self.translator.translate(
            original,
            source_language,
            self.target_language,
        )
        chosen_path = save_path
        if before_speech is not None:
            returned_path = before_speech(original, translated)
            if returned_path is not None:
                chosen_path = returned_path
        if self.use_tts and translated and self.speaker is not None:
            self.speaker.speak(translated, self.target_language, save_path=chosen_path)
        return original, translated


def listen_forever(
    recorder,
    phrase_translator: PhraseTranslator,
    playback_gap: float = 0.4,
    save_audio: bool = False,
    verbose: bool = False,
) -> None:
    """Keep translating phrases until the user presses Ctrl+C.

    The recorder is only called between phrases. ``process`` plays audio before
    this function asks for the next recording, so TTS is not fed back into ASR.
    """

    print("Listening...")
    print()
    while True:
        try:
            audio = recorder.record_utterance()
        except KeyboardInterrupt:
            print()
            print("Stopped.")
            return
        if audio is None:
            if verbose:
                print("(Ignored a short or quiet sound.)")
            continue
        saved_path: dict[str, Path | None] = {"path": None}

        def before_speech(original: str, translated: str):
            print(f"Original: {original}")
            print(f"Translated: {translated}")
            print()
            path = None
            if save_audio and translated:
                path = output_file_for(translated)
                saved_path["path"] = path
            if phrase_translator.use_tts and translated and phrase_translator.speaker is not None:
                print("Speaking translation...")
            return path

        try:
            original, _translated = phrase_translator.process(
                audio,
                before_speech=before_speech,
            )
            if not original:
                if verbose:
                    print("(Nothing was recognized.)")
                continue
            if saved_path["path"] is not None:
                print(f"Saved audio: {saved_path['path']}")
            if phrase_translator.use_tts and _translated and playback_gap:
                time.sleep(playback_gap)
                print()
        except KeyboardInterrupt:
            print()
            print("Stopped.")
            return
        except Exception as exc:
            print(f"Could not translate that phrase: {exc}")
            print()
        print("Listening...")
        print()


def download_required_models(args: argparse.Namespace) -> None:
    from asr import ASR_DOWNLOAD_BYTES, SpeechRecognizer
    from nmt import MODEL_ID, NMT_WEIGHT_BYTES, TextTranslator
    from tts import Speaker, voice_available, voice_for_language

    device = resolve_device(args.device)
    paths = prepare_model_directories(args.models_dir)
    asr_bytes = ASR_DOWNLOAD_BYTES[args.asr_model]
    print("Internet access is required for any model that is not already downloaded.")
    print(f"Whisper '{args.asr_model}' repository size: about {asr_bytes / 1_000_000:.0f} MB")
    print("The same Whisper model recognizes every supported spoken language.")
    print(
        f"Translation weights for {MODEL_ID}: {NMT_WEIGHT_BYTES / 1_000_000:.0f} MB "
        "plus tokenizer files"
    )
    print("That one translation model is reused for every language pair.")
    if voice_available(args.target):
        voice = voice_for_language(args.target, args.tts_voice)
        print(f"Piper voice {voice.voice_id}: {voice.size_bytes / 1_000_000:.0f} MB")
    else:
        print(
            f"No Piper voice will be downloaded for '{args.target}'. "
            "Translated text still works with --no-tts."
        )
    print("Models already on disk are not downloaded again.")
    print("Only the voice for the selected target language is downloaded.")
    print()
    print(f"Preparing Whisper ({args.asr_model}) in {paths['asr']}...")
    SpeechRecognizer(args.asr_model, paths["asr"], device).load()
    print(f"Preparing translation model {MODEL_ID} in {paths['nmt']}...")
    TextTranslator(paths["nmt"], device).load()
    if voice_available(args.target):
        voice = voice_for_language(args.target, args.tts_voice)
        print(f"Preparing Piper voice {voice.voice_id} in {paths['tts']}...")
        Speaker(
            voices_dir=paths["tts"],
            use_cuda=device == "cuda",
            voice_id=args.tts_voice,
        ).load(args.target)
    print()
    print("Models downloaded and loaded successfully.")
    print("The translation loop can now run locally, without sending audio to an API.")


def run_text(args: argparse.Namespace) -> int:
    from nmt import TextTranslator
    from tts import Speaker

    device = resolve_device(args.device)
    paths = prepare_model_directories(args.models_dir)
    text = " ".join(args.text.split())
    if not text:
        print("Nothing to translate.")
        return 0
    use_tts = not args.no_tts
    if use_tts:
        validate_audio_device(args.output_device, "output")
    output_label = "Not used (--no-tts)" if not use_tts else device_label(args.output_device, "output")
    print_banner(args.source, args.target, "Typed text", output_label)
    print("Loading models. Missing files are downloaded on first use, which can take several minutes.")
    print()
    translator = TextTranslator(paths["nmt"], device)
    if args.source == "auto":
        translator.load()
    else:
        translator.load(args.source, args.target)
    speaker = None
    if use_tts:
        speaker = Speaker(
            voices_dir=paths["tts"],
            output_device=args.output_device,
            use_cuda=device == "cuda",
            voice_id=args.tts_voice,
        )
        speaker.load(args.target)
    print("Models loaded successfully.")
    print()
    translated = translator.translate(text, args.source, args.target)
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    if use_tts and translated and speaker is not None:
        print()
        print("Speaking translation...")
        save_path = output_file_for(translated) if args.save_audio else None
        speaker.speak(translated, args.target, save_path=save_path)
        if save_path is not None:
            print(f"Saved audio: {save_path}")
    return 0


def run_microphone(args: argparse.Namespace) -> int:
    from asr import DetectorConfig, MicrophoneUtteranceRecorder, SpeechRecognizer
    from nmt import TextTranslator
    from tts import Speaker

    device = resolve_device(args.device)
    paths = prepare_model_directories(args.models_dir)
    use_tts = not args.no_tts
    validate_audio_device(args.input_device, "input")
    if use_tts:
        validate_audio_device(args.output_device, "output")
    output_label = "Not used (--no-tts)" if not use_tts else device_label(args.output_device, "output")
    print_banner(
        args.source,
        args.target,
        device_label(args.input_device, "input"),
        output_label,
    )
    print("Loading models. Missing files are downloaded on first use, which can take several minutes.")
    print()
    recognizer = SpeechRecognizer(args.asr_model, paths["asr"], device)
    recognizer.load()
    translator = TextTranslator(paths["nmt"], device)
    if args.source == "auto":
        translator.load()
    else:
        translator.load(args.source, args.target)
    speaker = None
    if use_tts:
        speaker = Speaker(
            voices_dir=paths["tts"],
            output_device=args.output_device,
            use_cuda=device == "cuda",
            voice_id=args.tts_voice,
        )
        speaker.load(args.target)
    config = DetectorConfig(
        threshold=args.speech_threshold if args.speech_threshold is not None else 0.015,
        silence_seconds=args.silence_duration,
        min_speech_seconds=args.min_speech_duration,
        max_speech_seconds=args.max_speech_duration,
    )
    recorder = MicrophoneUtteranceRecorder(args.input_device, config)
    if args.speech_threshold is None:
        print("Measuring ambient noise. Stay quiet for a moment...")
        try:
            threshold = recorder.calibrate()
        except Exception as exc:
            raise SystemExit(f"Could not calibrate the microphone: {exc}") from exc
        print(f"Calibrated speech threshold: {threshold:.4f}")
        print()
    print("Models loaded successfully.")
    print()
    phrase_translator = PhraseTranslator(
        recognizer,
        translator,
        speaker,
        args.source,
        args.target,
        use_tts=use_tts,
    )
    listen_forever(
        recorder,
        phrase_translator,
        playback_gap=args.playback_gap,
        save_audio=args.save_audio,
        verbose=args.verbose,
    )
    return 0


def validate_languages(args, auto, model_name_for, require_language, voice_available, voice_for_language) -> None:
    """Reject language choices before any model download starts."""

    from nmt import UnsupportedLanguagePair

    if args.target == auto:
        raise UnsupportedLanguagePair(
            "--target cannot be auto. Choose the language you want to hear."
        )
    require_language(args.target)
    if args.source == auto:
        if args.text is not None and not args.download_models:
            raise UnsupportedLanguagePair(
                "Automatic language detection listens to the microphone. "
                "For typed text, set --source to the language of that text."
            )
    else:
        model_name_for(args.source, args.target)
    needs_voice = (not args.no_tts) or bool(args.tts_voice)
    if args.download_models and not voice_available(args.target) and not args.tts_voice:
        return
    if needs_voice:
        voice_for_language(args.target, args.tts_voice)


def warn_python_version() -> None:
    version = sys.version_info
    if version < (3, 10):
        raise SystemExit(
            "Python 3.10 or newer is required. Install Python 3.12 from "
            "https://www.python.org/downloads/windows/ and run: py -3.12 -m venv .venv"
        )
    if version >= (3, 13):
        print(
            "Warning: this project was tested with Python 3.12. "
            f"You are using Python {version.major}.{version.minor}. "
            "Install Python 3.12 and recreate the virtual environment with: py -3.12 -m venv .venv"
        )
        print()


def main(argv: list[str] | None = None) -> int:
    configure_console()
    warn_python_version()
    args = build_parser().parse_args(argv)
    validate_runtime_arguments(args)
    if args.list_devices:
        list_audio_devices()
        return 0
    if args.list_languages:
        list_supported_languages()
        return 0

    try:
        from asr import AUTO_LANGUAGE, AudioCaptureError
        from nmt import TranslationError, UnsupportedLanguagePair, model_name_for, require_language
        from tts import SpeechSynthesisError, voice_available, voice_for_language
    except ImportError as exc:
        raise SystemExit(missing_dependency_message(exc)) from exc

    try:
        validate_languages(args, AUTO_LANGUAGE, model_name_for, require_language, voice_available, voice_for_language)
    except (UnsupportedLanguagePair, SpeechSynthesisError) as exc:
        print(exc)
        return 2

    try:
        if args.download_models:
            download_required_models(args)
            return 0
        if args.text is not None:
            return run_text(args)
        return run_microphone(args)
    except KeyboardInterrupt:
        print()
        print("Stopped.")
        return 0
    except (AudioCaptureError, TranslationError, SpeechSynthesisError) as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
