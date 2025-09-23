!pip install --upgrade pip
!pip install transformers accelerate datasets trl peft bitsandbytes huggingface_hub comet_ml unsloth

import os
import torch
from datasets import load_dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOConfig, DPOTrainer
from huggingface_hub import HfApi, login

# === 1. Configuration ===
model_name = "asulova/hamlet-merged"   # Already merged SFT model
max_seq_length = 2048
output_dir = "./dpo_hamlet"

# Optional: CometML logging
os.environ["COMET_API_KEY"] = ""  # or leave empty
os.environ["COMET_PROJECT_NAME"] = "shakespeare"
os.environ["COMET_WORKSPACE"] = "aldi-sulova"

# === 2. Load model and tokenizer ===
device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
)

# === 3. Apply LoRA adapters for DPO ===
# This creates temporary adapters on top of the merged model.
# After training, weights will be saved fully merged.
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
)

# === 4. Load preference dataset ===
dataset = load_dataset("json", data_files="hamlet_dpo.jsonl", split="train")

# === 5. Format samples ===
alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
EOS_TOKEN = tokenizer.eos_token

def format_samples(example):
    example["prompt"] = alpaca_template.format(example["instruction"])
    example["chosen"] = example["chosen"] + EOS_TOKEN
    example["rejected"] = example["rejected"] + EOS_TOKEN
    return {"prompt": example["prompt"], "chosen": example["chosen"], "rejected": example["rejected"]}

dataset = dataset.map(format_samples)
dataset = dataset.train_test_split(test_size=0.05)

# === 6. Configure DPO Trainer ===
training_args = DPOConfig(
    learning_rate=2e-6,
    lr_scheduler_type="linear",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    warmup_steps=10,
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=50,          # integer, not fraction
    logging_steps=1,
    report_to="comet_ml",   # set "none" if not using Comet
    seed=0,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    beta=0.5,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_length=max_seq_length // 2,
    max_prompt_length=max_seq_length // 2,
    args=training_args,
)

# === 7. Start DPO training ===
trainer.train()

# === 8. Save merged model ===
# Since hamlet-merged was already a merged SFT model, saving here
# will produce a new standalone model with DPO weights included.
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# === 9. Push to Hugging Face Hub ===
login(token="your_hf_token_here")   # replace with your token or set HF_TOKEN env var

api = HfApi()
api.upload_folder(
    folder_path=output_dir,
    repo_id="asulova/hamlet-dpo",
    repo_type="model",
)

# === 10. Sanity check / inference ===
FastLanguageModel.for_inference(model)
test_prompt = alpaca_template.format("Explain Polonius' advice to Laertes about friendship.")
inputs = tokenizer([test_prompt], return_tensors="pt").to(device)
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True)
