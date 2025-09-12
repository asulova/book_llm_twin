import os
import re
import json
import requests
import logging
from typing import List, Tuple
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
class InstructionAnswerSet:
    """Class to hold a set of instruction-answer pairs."""

    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    @classmethod
    def from_jsonl_flexible(cls, jsonl_text: str) -> 'InstructionAnswerSet':
        """Parse JSONL text and recover instruction-answer pairs even if output is noisy."""
        pairs = []
        for line in jsonl_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove common fences/backticks
            line = re.sub(r"^```json|```$", "", line).strip()
            try:
                obj = json.loads(line)
                # Case 1: direct instruction-answer
                if "instruction" in obj and "answer" in obj:
                    pairs.append((obj["instruction"], obj["answer"]))
                    continue

                # Case 2: Ollama wrapped response
                if "response" in obj:
                    try:
                        inner_raw = obj["response"]
                        inner = json.loads(inner_raw) if isinstance(inner_raw, str) else inner_raw
                        # Inner container with multiple pairs
                        if isinstance(inner, dict) and isinstance(inner.get("instruction_answer_pairs"), list):
                            for item in inner["instruction_answer_pairs"]:
                                if (
                                    isinstance(item, dict)
                                    and "instruction" in item and "answer" in item
                                ):
                                    pairs.append((item["instruction"], item["answer"]))
                            continue
                        # Inner single pair
                        if isinstance(inner, dict) and "instruction" in inner and "answer" in inner:
                            pairs.append((inner["instruction"], inner["answer"]))
                            continue
                    except json.JSONDecodeError:
                        pass
            except json.JSONDecodeError:
                continue
        logger.info(f"Recovered {len(pairs)} instruction-answer pairs from model response.")
        return cls(pairs)

    def __iter__(self):
        return iter(self.pairs)

# -------------------------
# Helpers
# -------------------------
def clean_text(text: str) -> str:
    """Remove unwanted characters and extra whitespace."""
    text = re.sub(r"[^\w\s.,!?'\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_substrings(dataset: dict, min_length: int = 500, max_length: int = 1000) -> List[str]:
    """Split text into chunks using sentence boundaries."""
    extracts = []
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

def generate_instruction_answer_pairs(text: str, num_pairs: int = 5) -> InstructionAnswerSet:
    """Generate instruction-answer pairs from text using Ollama (non-streamed, JSON safe)."""
    prompt = f"""
        You are a JSON generator.

        Based on the following text, generate {num_pairs} instruction-answer pairs.  

        Rules:
        - Each instruction must ask about a specific theme, event, character, or idea present in the text.  
        - Use only concepts found in the text.  
        - Do NOT mention the text, context, system, or course. Instructions must be self-contained.  
        - Answers must imitate the style and tone of the text.  
        - The answer must be a coherent paragraph (4–6 sentences) in Shakespearean style:
        - Use "thee", "thou", "thy", "doth", "hath", etc.
        - Include dramatic and poetic phrasing.
        
        Provide your response in JSON format with the following structure:
        {{
        "instruction_answer_pairs": [
            {{"instruction": "...", "answer": "..."}},
            ...
            ]
        }}

        Example (Shakespeare style):       
        {{
        "instruction_answer_pairs": [
            {{"instruction": "What is Hamlet's grief over his father's death?", "answer": "O, how my soul is torn! The death of my father doth weigh upon me, heavier than the leaden sky. My heart mourns with ceaseless sorrow, and my eyes weep in secret for the lost guidance of my sire. Even the sun seems dimmed by my desolate thoughts, and all the world appears but a shadow of its former grace."}},
            {{"instruction": "Describe the ghost that appears to the sentinels.", "answer": "A specter clad in armour of ancient wars doth stride upon the battlements, silent and solemn. Its visage mirrors that of the late king, stern and regal, yet laden with silent grief. The sentinels quail before this ethereal form, hearts hammering as fear and wonder entwine in their souls."}},
            {{"instruction": "What advice does Laertes give to Ophelia?", "answer": "Mark well, sweet sister, the counsel I impart: keep thy heart guarded, and let not thy love be spent upon empty words. Regard the world with cautious eye, and let virtue guide thy steps. Be wary of entanglements with the prince, for the court's snares are deep, and innocence alone doth not suffice to shield thee."}},
            {{"instruction": "Describe Hamlet's internal conflict about avenging his father.", "answer": "O, to strike or to ponder, that is my torment! The treachery of my uncle burns within my breast, yet conscience restrains my hand. I am ensnared in thought, wavering 'twixt righteous vengeance and fear of immortal consequence. The night itself seems to echo my uncertainty, whispering doubts into the hollow chambers of my heart."}},
            {{"instruction": "What is the political tension in Denmark at the beginning of the play?", "answer": "Denmark stands wary, for young Fortinbras of Norway doth stir, seeking recompense for lands lost by his father. The kingdom is vigilant, as soldiers patrol the ramparts and sentinels keep watch in the chill of night. Tension clings to every stone of Elsinore, and whispers of war mingle with the cold wind that sweeps the castle battlements."}},
            ...
            ]
        }}
        Text:
        {text}
        """

    logger.info(f"Sending request to Ollama model: {OLLAMA_MODEL}")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL, 
            "prompt": prompt, 
            "format": "json",
            "stream": False},
        stream=False
    )

    if response.status_code != 200:
        logger.error(f"Request failed: {response.status_code}: {response.text}")
        raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

    raw_output = response.text.strip()
    logger.info(raw_output)

    return InstructionAnswerSet.from_jsonl_flexible(raw_output)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    input_file = "data/raw/hamlet.txt"
    output_file = "data/processed/hamlet_instructions.jsonl"

    logger.info(f"Reading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = {"content": [text]}
    logger.info("Splitting text into chunks...")
    chunks = extract_substrings(dataset)
    logger.info(f"Generated {len(chunks)} chunks.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"Processing chunk {idx}/{len(chunks)}")
            ia_set = generate_instruction_answer_pairs(chunk, num_pairs=10)

            for instruction, answer in ia_set:
                logger.info(f"Writing instruction-answer pair to file: {instruction}") 
                logger.info(f"Answer: {answer}")
                f_out.write(json.dumps({"instruction": instruction, "answer": answer}, ensure_ascii=False) + "\n")

    logger.info("Process completed successfully")
