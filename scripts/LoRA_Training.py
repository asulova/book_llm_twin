# finetune_qwen25_lora.py
# Fine-tune Qwen2.5-3B (base) with LoRA adapters (TRL 0.7+) and push to Hugging Face Hub

# === 0. Install dependencies ===
!pip install --upgrade pip
!pip install transformers accelerate datasets trl peft bitsandbytes huggingface_hub comet_ml

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from huggingface_hub import login, create_repo

# === 1. Hugging Face token & repo config ===
HF_TOKEN = ""  # <-- replace with your HF token
HF_REPO  = "asulova/hamlet-model"                     # <-- your repo name

# Optional: CometML logging
os.environ["COMET_API_KEY"] = ""  # or leave empty
os.environ["COMET_PROJECT_NAME"] = "shakespeare"
os.environ["COMET_WORKSPACE"] = "aldi-sulova"

# === 2. Load Qwen2.5-3B model ===
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # important for batching

# === 3. LoRA adapter config ===
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# === 4. Load dataset ===
print("📂 Upload your JSON dataset (fields: 'instruction' and 'answer')")
try:
    from google.colab import files
    uploaded = files.upload()
    dataset_filename = next(iter(uploaded))
except ImportError:
    dataset_filename = input("Enter path to your dataset JSON file: ")

dataset = load_dataset("json", data_files=dataset_filename, split="train")

def format_prompts(examples):
    texts = [f"### Instruction:\n{inst}\n\n### Answer:\n{ans}"
             for inst, ans in zip(examples["instruction"], examples["answer"])]
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    num_train_epochs=10,
    learning_rate=3e-4,
    lr_scheduler_type="linear",
    fp16=True,
    bf16=False,
    logging_steps=20,
    save_strategy="epoch",
    output_dir="./outputs",
    optim="adamw_8bit",
    seed=0,
    max_grad_norm=1.0,
    report_to="comet_ml",
)

# === 6. Initialize trainer ===
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

# === 7. Train LoRA adapters ===
trainer.train()

