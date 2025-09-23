from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_repo = "asulova/hamlet-model"

# Load base model on CPU
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_repo)
model.eval()

# Run inference
prompt = (
    """
INSTRUCTIONS:
Do not invent content, add ideas, or change facts.
Preserve the original meaning exactly; only adjust wording, phrasing, and style.

PASSAGE:
The sun set over the village, and the villagers gathered in the square to celebrate the harvest. Children laughed as they ran between the stalls, and the air was filled with music and the scent of fresh bread.
QUESTION:
Describe the celebration in a village during harvest time.

ANSWER:
"""
)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
