import os
import re
import json
import logging
from typing import Dict, Iterable, Optional, List, Tuple

import requests
from dotenv import load_dotenv


# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# -------------------------
# Environment variables
# -------------------------
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_INSTRUCT_MODEL", "llama3")


# -------------------------
# Classes
# -------------------------
class DpoSet:
    """Container for DPO triples: (instruction, chosen, rejected)."""

    def __init__(self, pairs: List[Tuple[str, str, str]]):
        self.pairs = pairs

    def __iter__(self):
        for instruction, chosen, rejected in self.pairs:
            yield {"instruction": instruction, "chosen": chosen, "rejected": rejected}

    @classmethod
    def from_jsonl_flexible(cls, raw_text: str) -> "DpoSet":
        """Parse Ollama output robustly.

        Expected core structures:
        - {"pairs": [{"instruction": str, "chosen": str, "rejected": str}, ...]}
        - or wrapped under {"response": "...json..."}
        - tolerate ```json fences
        """
        text = raw_text.strip()
        text = re.sub(r"^```json|```$", "", text).strip()
        pairs: List[Tuple[str, str, str]] = []

        def _consume(obj: Dict) -> None:
            nonlocal pairs
            items = obj.get("pairs")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    p = (it.get("instruction") or "").strip()
                    c = (it.get("chosen") or "").strip()
                    r = (it.get("rejected") or "").strip()
                    if p and c and r:
                        pairs.append((p, c, r))

        # Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                if "response" in obj:
                    inner_raw = obj["response"]
                    try:
                        inner = json.loads(inner_raw) if isinstance(inner_raw, str) else inner_raw
                        if isinstance(inner, dict):
                            _consume(inner)
                    except json.JSONDecodeError:
                        pass
                else:
                    _consume(obj)
        except json.JSONDecodeError:
            pass

        logging.info(f"Recovered {len(pairs)} DPO pairs from model response.")
        return cls(pairs)


# -------------------------
# Helpers
# -------------------------
def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_substrings(dataset: dict, min_length: int = 500, max_length: int = 1000) -> List[str]:
    extracts: List[str] = []
    # Match SFT script's sentence boundary rule exactly
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for article in dataset.get("content", []):
        cleaned_article = clean_text(article)
        sentences = re.split(sentence_pattern, cleaned_article)
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
                current_chunk = sentence + " "
        if len(current_chunk) >= min_length:
            extracts.append(current_chunk.strip())
    return extracts


def generate_dpo_pairs_from_text(text: str, num_pairs: int = 10) -> List[Dict[str, str]]:
    """Ask Ollama to produce multiple DPO pairs from a text chunk.

    Returns a list of objects with keys: prompt, chosen, rejected.
    """
    prompt = f"""
        You are a JSON generator.

        Based on the following Shakespearean text, generate {num_pairs} items. For each item:
        - "instruction": An instruction about a specific theme, event, character, or idea present in the text. The prompt must be self-contained and not reference this task or the chunk.
        - "chosen": A high-quality Shakespearean answer (4–6 sentences) using archaic style (thee, thou, thy, doth, hath), dramatic tone, and poetic phrasing.
        - "rejected": Offer high-quality Shakespearean answer, three to five lines, wrought in like Shakespearean style, yet astray from truth or fact.  

        Rules:
        - Use only concepts present in the text.
        - Do not include additional commentary.
        - Return ONLY valid JSON with the structure:
        {{
            "pairs": [
            {{"instruction": "...", "chosen": "...", "rejected": "..."}},
            ...
            ]
        }}

        Text:
        {text}

        """

    logger.info(f"Requesting DPO pairs from Ollama model: {OLLAMA_MODEL}")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        },
        stream=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

    raw_output = response.text.strip()
    dpo_set = DpoSet.from_jsonl_flexible(raw_output)
    return list(dpo_set)


def build_dpo_rows_from_raw_text(raw_text_path: str, num_pairs_per_chunk: int = 8) -> Iterable[Dict]:
    with open(raw_text_path, "r", encoding="utf-8") as f:
        text = f.read()
    dataset = {"content": [text]}
    chunks = extract_substrings(dataset)
    logger.info(f"Generated {len(chunks)} chunks from raw text.")
    for idx, chunk in enumerate(chunks, start=1):
        logger.info(f"Processing chunk {idx}/{len(chunks)}")
        rows = generate_dpo_pairs_from_text(chunk, num_pairs=num_pairs_per_chunk)
        for row in rows:
            yield row


def main() -> None:
    # Default configuration (no CLI args)
    RAW_TEXT_PATH = "data/raw/hamlet.txt"
    OUTPUT_PATH = "data/processed/hamlet_dpo.jsonl"
    PAIRS_PER_CHUNK = 10
    LIMIT: Optional[int] = None  # Set to an int to cap total rows

    logger.info(f"Reading raw text from: {RAW_TEXT_PATH}")
    total = 0
    out_path = OUTPUT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for row in build_dpo_rows_from_raw_text(RAW_TEXT_PATH, num_pairs_per_chunk=PAIRS_PER_CHUNK):
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()
            logger.info(f"Wrote DPO item: prompt={row['instruction'][:60]}... | chosen_len={len(row['chosen'])} | rejected_len={len(row['rejected'])}")
            total += 1
            if LIMIT is not None and total >= LIMIT:
                break
    logger.info(f"Wrote {total} DPO examples to {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()


