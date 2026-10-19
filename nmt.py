# Copyright (C) 2026 Nominally
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multilingual translation with one cached M2M100 model.

The 418M Facebook model translates between any pair of its 100 languages, so
the program does not download a separate model for each direction. Weights are
the PyTorch file only. Transformers would otherwise start a second safetensors
download that this loader never reads.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

# Marian-style tokenizers are not used. The filter stays so an older cached
# warning cannot leak into the tutorial output if Transformers prints it.
warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses.")
# The published M2M100 generation config includes early_stopping. This
# Transformers build ignores that flag and would otherwise print a warning
# on every sentence. The flag is left unchanged because setting it to None
# makes generation reject the config.
warnings.filterwarnings("ignore", message="The following generation flags are not valid")
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

MODEL_ID = "facebook/m2m100_418M"
# pytorch_model.bin on the model card. The repository also contains a Rust
# weight file of about the same size. This loader does not request that file.
NMT_WEIGHT_BYTES = 1_935_796_948

# Language codes published on the facebook/m2m100_418M model card (MIT).
M2M100_LANGUAGE_CODES = frozenset(
    {
        "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br",
        "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es",
        "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu",
        "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo",
        "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb",
        "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr",
        "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl",
        "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq",
        "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr",
        "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu",
    }
)

# Languages called out in the tutorial. Other M2M100 codes still translate.
TUTORIAL_LANGUAGES = (
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("de", "German"),
    ("ko", "Korean"),
    ("zh", "Chinese"),
)
LANGUAGE_NAMES = dict(TUTORIAL_LANGUAGES)


class UnsupportedLanguagePair(ValueError):
    """The requested language code is not in the M2M100 set, or the pair is empty."""


class TranslationError(RuntimeError):
    """The model was selected, but inference did not produce a translation."""


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def supported_pairs() -> list[tuple[str, str]]:
    """Every ordered pair of tutorial languages. The model is still one file."""

    codes = [code for code, _name in TUTORIAL_LANGUAGES]
    return [(source, target) for source in codes for target in codes if source != target]


def format_supported_pairs() -> str:
    return ", ".join(f"{source} -> {target}" for source, target in supported_pairs())


def require_language(code: str) -> str:
    if code not in M2M100_LANGUAGE_CODES:
        raise UnsupportedLanguagePair(
            f"Unsupported language '{code}'. "
            "Run --list-languages to see the tutorial languages and the other "
            "M2M100 codes this model can translate."
        )
    return code


def model_name_for(source_language: str, target_language: str) -> str:
    """Return the single multilingual model id after checking both codes."""

    if source_language == target_language:
        raise UnsupportedLanguagePair(
            "Source and target languages must be different. "
            f"Both were '{source_language}'."
        )
    require_language(source_language)
    require_language(target_language)
    return MODEL_ID


def load_pretrained(cache_dir: Path) -> tuple[object, object]:
    """Download the PyTorch weights once and return the tokenizer and model."""

    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    tokenizer = M2M100Tokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(cache_dir),
    )
    # use_safetensors=False keeps the download on pytorch_model.bin.
    # The model card has no safetensors file. The env var above stops
    # the background conversion that would save a second copy.
    model = M2M100ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=str(cache_dir),
        use_safetensors=False,
    )
    return tokenizer, model


class TextTranslator:
    """Load M2M100 once and reuse it for every sentence and language pair."""

    def __init__(self, cache_dir: Path | str = "models/nmt", device: str = "cpu") -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("Translation device must be 'cpu' or 'cuda'.")
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._tokenizer = None
        self._model = None

    def load(self, source_language: str | None = None, target_language: str | None = None) -> None:
        """Download the model if needed, then keep it in memory.

        A source and target are checked when both are given. The weights do
        not depend on the direction.
        """

        if source_language is not None and target_language is not None:
            model_name_for(source_language, target_language)
        self._ensure_model()

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """Translate one phrase. Empty input returns an empty string."""

        cleaned = " ".join(text.split())
        if not cleaned:
            return ""

        model_name_for(source_language, target_language)
        try:
            tokenizer, model = self._ensure_model()
            # src_lang selects the encoder language token. forced_bos selects
            # the decoder language. Whisper is not asked to translate.
            tokenizer.src_lang = source_language
            encoded = tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            target_token = tokenizer.get_lang_id(target_language)
            import torch

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=target_token,
                    max_new_tokens=256,
                    num_beams=1,
                )
            decoded = tokenizer.decode(generated[0].detach().cpu(), skip_special_tokens=True)
        except (UnsupportedLanguagePair, TranslationError):
            raise
        except Exception as exc:
            raise TranslationError(f"Translation failed ({MODEL_ID}): {exc}") from exc
        return " ".join(decoded.split())

    def _ensure_model(self) -> tuple[object, object]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading translation model {MODEL_ID}...")
        try:
            tokenizer, model = load_pretrained(self.cache_dir)
            model.to(self.device)
            model.eval()
        except ImportError as exc:
            raise TranslationError(
                "transformers is not installed. Install requirements before translating."
            ) from exc
        except Exception as exc:
            raise TranslationError(
                "Could not download or load the translation model. An internet "
                "connection is required the first time. Interrupted Hugging Face "
                f"downloads can be resumed by running the command again. Details: {exc}"
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model
