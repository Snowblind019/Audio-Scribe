"""Translating the transcript into Romanian or English, offline.

Uses NLLB-200 (Meta's "No Language Left Behind" model, the distilled 600M version) run by
CTranslate2, which is already installed for Whisper, so no new packages are needed. The
model (about 640 MB) downloads from Hugging Face the first time and is checked:
  * it is pinned to one exact revision of the repository, and
  * every file is compared with the SHA-256 (or git) hash written below before it is used.
CTranslate2 model files are plain weights, not code.

NLLB is licensed CC-BY-NC 4.0: fine for personal use, not for selling translations.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

MODEL_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
MODEL_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
# file -> ("sha256", hash) for large files, ("git", blob hash) for small ones
MODEL_FILES = {
    "model.bin": ("sha256", "ed1beaf75134de7505315a5223162f56acff397eff6b50638a500d3936fe707b"),
    "tokenizer.json": ("sha256", "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"),
    "config.json": ("git", "338d32248dd94e6268192c8354c41c0eca5eeb62"),
    "shared_vocabulary.txt": ("git", "1f831354dfc504f6b866ad05aa8efd94919badf9"),
}
MODEL_SIZE_MB = 640

TARGETS = [("ro", "Romanian"), ("en", "English")]

# Whisper language codes -> NLLB codes
NLLB = {
    "en": "eng_Latn", "ro": "ron_Latn", "ar": "arb_Arab", "zh": "zho_Hans", "cs": "ces_Latn", "nl": "nld_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "el": "ell_Grek", "hi": "hin_Deva", "hu": "hun_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "pl": "pol_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "es": "spa_Latn",
    "sv": "swe_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "bg": "bul_Cyrl", "ca": "cat_Latn", "da": "dan_Latn",
    "fi": "fin_Latn", "he": "heb_Hebr", "hr": "hrv_Latn", "id": "ind_Latn", "lt": "lit_Latn", "lv": "lvs_Latn",
    "ms": "zsm_Latn", "no": "nob_Latn", "fa": "pes_Arab", "sk": "slk_Latn", "sl": "slv_Latn", "sr": "srp_Cyrl",
    "th": "tha_Thai", "vi": "vie_Latn", "et": "est_Latn", "ur": "urd_Arab", "bn": "ben_Beng", "ta": "tam_Taml",
    "mk": "mkd_Cyrl", "be": "bel_Cyrl", "sq": "als_Latn", "hy": "hye_Armn", "ka": "kat_Geor", "az": "azj_Latn",
    "kk": "kaz_Cyrl", "af": "afr_Latn", "sw": "swh_Latn", "tl": "tgl_Latn", "is": "isl_Latn", "cy": "cym_Latn",
}


class Stopped(Exception):
    pass


class ModelProblem(RuntimeError):
    pass


def _git_blob_sha1(path: Path) -> str:
    """The git id of a file. Hugging Face lists small files by this id only. It ties the file
    to the pinned revision; SHA-1 still resists making a second file with the same id."""
    h = hashlib.sha1()  # nosec B324 - checks a file against the pinned git revision
    h.update(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _verified(path: Path, kind: str, expected: str) -> bool:
    actual = _sha256(path) if kind == "sha256" else _git_blob_sha1(path)
    return actual == expected


def model_dir(report: Callable[[str], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> Path:
    """Downloads (once) and checks the translation model. Returns the folder it is in."""
    from huggingface_hub import hf_hub_download

    folder = None
    for i, (name, (kind, expected)) in enumerate(MODEL_FILES.items()):
        if should_stop and should_stop():
            raise Stopped()
        if report:
            report(f"{i + 1}/{len(MODEL_FILES)} {name}")
        path = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION))
        if not _verified(path, kind, expected):
            # Remove the bad copy (and the cached file behind the link) so the next try downloads it again.
            for bad in {path.resolve(), path}:
                try:
                    os.remove(bad)
                except OSError:
                    pass
            raise ModelProblem(f"The translation model file {name} failed its hash check, so it was not used.")
        if folder is None:
            folder = path.parent
        elif path.parent != folder:
            raise ModelProblem("The translation model files were not saved together.")
    return folder


class Translator:
    def __init__(self, folder: Path, device: str = "cpu"):
        import ctranslate2
        from tokenizers import Tokenizer

        threads = max(2, (os.cpu_count() or 4) // 2)
        self.model = ctranslate2.Translator(str(folder), device=device, compute_type="int8",
                                            intra_threads=threads)
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))

    def translate(self, lines: list[str], source: str, target: str,
                  progress: Callable[[float], None] | None = None,
                  should_stop: Callable[[], bool] | None = None, batch: int = 16) -> list[str]:
        """Translate each line. source and target are Whisper codes like "en" or "ro"."""
        src, tgt = NLLB.get(source), NLLB.get(target)
        if not src or not tgt:
            raise ModelProblem(f"Translating from {source} isn't supported.")
        out: list[str] = []
        for i in range(0, len(lines), batch):
            if should_stop and should_stop():
                raise Stopped()
            chunk = lines[i:i + batch]
            tokens = [[src] + self.tok.encode(text, add_special_tokens=False).tokens + ["</s>"] for text in chunk]
            results = self.model.translate_batch(tokens, target_prefix=[[tgt]] * len(chunk), beam_size=4,
                                                 max_decoding_length=256, repetition_penalty=1.1)
            for text, res in zip(chunk, results):
                if not text.strip():
                    out.append("")
                    continue
                pieces = [p for p in res.hypotheses[0] if p not in (tgt, "</s>", "<s>")]
                ids = [self.tok.token_to_id(p) for p in pieces]
                out.append(self.tok.decode([x for x in ids if x is not None], skip_special_tokens=True).strip())
            if progress:
                progress(min(1.0, (i + len(chunk)) / max(1, len(lines))))
        return out
