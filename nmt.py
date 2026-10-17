# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neural machine translation with cached MarianMT models.

Only language pairs listed in ``TRANSLATION_MODELS`` are offered. Add a new
pair after you have verified that its Hugging Face model actually loads and
produces the language you expect.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

# MarianTokenizer suggests sacremoses, but these Opus-MT models load and
# translate with the bundled SentencePiece files. The extra package is not used.
warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses.")

# These two Helsinki-NLP models were checked on their Hugging Face model cards
# (Apache-2.0). Each pytorch_model.bin file is 312,087,523 bytes. Tokenizer
# files add a few megabytes. TensorFlow and Flax copies in the same repositories
# are not downloaded by this project. Transformers would otherwise start a
# background download of a second safetensors copy that this loader never reads.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
TRANSLATION_MODELS = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
}
NMT_WEIGHT_BYTES = 312_087_523


class UnsupportedLanguagePair(ValueError):
    """The requested translation direction has no verified model."""


class TranslationError(RuntimeError):
    """The model was selected, but inference did not produce a translation."""


def supported_pairs() -> list[tuple[str, str]]:
    return list(TRANSLATION_MODELS)


def format_supported_pairs() -> str:
    return ", ".join(f"{source} -> {target}" for source, target in supported_pairs())


def model_name_for(source_language: str, target_language: str) -> str:
    """Return the Hugging Face model id for a verified language pair."""

    try:
        return TRANSLATION_MODELS[(source_language, target_language)]
    except KeyError as exc:
        raise UnsupportedLanguagePair(
            f"Unsupported language pair: {source_language} -> {target_language}. "
            f"Verified pairs: {format_supported_pairs()}. "
            "Add a model to TRANSLATION_MODELS in nmt.py only after you have tested it."
        ) from exc


class TextTranslator:
    """Load one Marian model per direction and reuse it."""

    def __init__(self, cache_dir: Path | str = "models/nmt", device: str = "cpu") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("Translation device must be 'cpu' or 'cuda'.")
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._cache: dict[tuple[str, str], tuple[object, object]] = {}

    def load(self, source_language: str, target_language: str) -> None:
        """Download the model if needed, then keep it in memory."""

        self._get(source_language, target_language)

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate one phrase. Empty input returns an empty string."""

        cleaned = " ".join(text.split())
        if not cleaned:
            return ""

        model_name = model_name_for(source_language, target_language)
        try:
            tokenizer, model = self._get(source_language, target_language)
            # Direct generation avoids the translation pipeline helper, which
            # has changed across Transformers releases.
            encoded = tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            import torch

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=256,
                    num_beams=1,
                )
            decoded = tokenizer.decode(generated[0].detach().cpu(), skip_special_tokens=True)
        except (UnsupportedLanguagePair, TranslationError):
            raise
        except Exception as exc:
            raise TranslationError(
                f"Translation failed ({model_name}): {exc}"
            ) from exc
        return " ".join(decoded.split())

    def _get(self, source_language: str, target_language: str) -> tuple[object, object]:
        key = (source_language, target_language)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        model_name = model_name_for(source_language, target_language)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading translation model {model_name}...")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError(
                "transformers is not installed. Install requirements before translating."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
            )
            model.to(self.device)
            model.eval()
        except Exception as exc:
            raise TranslationError(
                "Could not download or load the translation model. An internet "
                "connection is required the first time. Interrupted Hugging Face "
                f"downloads can be resumed by running the command again. Details: {exc}"
            ) from exc

        loaded = (tokenizer, model)
        self._cache[key] = loaded
        return loaded
