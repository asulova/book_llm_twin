from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configure base model
base_model_name = os.getenv("BASE_MODEL", "asulova/hamlet-merged")

# Load base model on CPU
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Run inference with a strict grounded prompt (demo)
prompt = (
    """
INSTRUCTIONS:
Do not invent content, add ideas, or change facts.
Preserve the original meaning exactly; only adjust wording, phrasing, and style according to the model's training.

PASSAGE:
Polonius advises Laertes to carefully choose his friends and to hold fast to the worthy ones. He says to "grapple them unto thy soul with hoops of steel," meaning to bind true friends closely and steadfastly. He also warns against spreading oneself thin with every new or unproven acquaintance, and counsels familiarity without vulgarity.

QUESTION:
What guidance does Polonius give Laertes about choosing and keeping friends?

ANSWER:
"""
)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
