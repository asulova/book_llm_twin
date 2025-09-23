import os
import sys
from typing import List, Dict

# Ensure project root is on PYTHONPATH so `rag` can be imported when running from `app/`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

import streamlit as st
import logging
from dotenv import load_dotenv

from rag.retriever import QdrantRetriever
from rag.generator import generate_answer_stream
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from rag.prompt import build_prompt, build_style_prompt


load_dotenv()


st.set_page_config(page_title="Book LLM Twin", page_icon="📚", layout="wide")
st.title("📚 Book LLM Twin")
st.caption("Local RAG using Qdrant, Sentence-Transformers, and Ollama")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

with st.sidebar:
	st.header("Settings")

	collection_name = st.text_input("Qdrant collection", value=os.getenv("QDRANT_COLLECTION", ""))

	# Embedding model is fixed via .env; no UI selection
	embedding_model = os.getenv("EMBEDDING_MODEL", "")
	if not embedding_model:
		st.warning("Set EMBEDDING_MODEL in your .env file")
	else:
		st.caption(f"Embedding model (from .env): {embedding_model}")

	# Ollama model is fixed via .env; no UI selection
	ollama_model = os.getenv("OLLAMA_MODEL", "")
	if not ollama_model:
		st.warning("Set OLLAMA_MODEL in your .env file")
	else:
		st.caption(f"Ollama model (from .env): {ollama_model}")

	top_k = st.slider("Top-K (retrieval)", 1, 10, int(os.getenv("TOP_K", "3")))
	score_threshold = st.slider("Score threshold (optional)", 0.0, 1.0, 0.0, 0.01)

	st.divider()
	st.subheader("Generation")
	num_predict = st.slider("Max tokens (num_predict)", 32, 1024, int(os.getenv("OLLAMA_NUM_PREDICT", "256")), 32)
	temperature = st.slider("Temperature", 0.0, 1.5, float(os.getenv("OLLAMA_TEMPERATURE", "0.2")), 0.05)
	# Always stream tokens in this simplified version
	stream_mode = True

	st.subheader("Prompt limits")
	prompt_max_contexts = st.number_input("Max contexts", 1, 10, int(os.getenv("PROMPT_MAX_CONTEXTS", "4")))
	prompt_max_chars = st.number_input("Max chars per context", 200, 5000, int(os.getenv("PROMPT_MAX_CHARS_PER_CONTEXT", "1200")), 50)

query = st.text_area("Enter your question (English):", height=120)
ask = st.button("Ask")


def _to_ctx(retrieved) -> List[Dict[str, str]]:
	return [{"text": r.text, "source": r.source} for r in retrieved]


if ask and query.strip():
	status = st.status("Contacting Qdrant and searching...", state="running")
	status.write("Initializing retrieval...")
	try:
		retriever = QdrantRetriever(collection_name=collection_name, embedding_model_name=embedding_model, top_k=top_k)
		status.write("Querying Qdrant for top-K passages...")
		hits = retriever.retrieve(query=query, score_threshold=(score_threshold or None))
		status.update(label="Qdrant search completed", state="complete")
	except Exception as e:
		status.update(label=f"Qdrant search failed: {e}", state="error")
		raise

	st.subheader("Modern-English Answer (Base Model via Ollama)")
	ctx = _to_ctx(hits)
	os.environ["PROMPT_MAX_CONTEXTS"] = str(int(prompt_max_contexts))
	os.environ["PROMPT_MAX_CHARS_PER_CONTEXT"] = str(int(prompt_max_chars))
	os.environ["OLLAMA_NUM_PREDICT"] = str(int(num_predict))
	os.environ["OLLAMA_TEMPERATURE"] = str(float(temperature))

	# Step 1 with Ollama: stream and collect the grounded modern-English answer
	gen_status = st.status("Generating with Ollama...", state="running")
	modern_answer_chunks = []
	modern_answer_placeholder = st.empty()
	for idx, chunk in enumerate(generate_answer_stream(
		question=query,
		contexts=ctx,
		model=(ollama_model or None),
		max_tokens=int(num_predict),
		temperature=float(temperature),
	)):
		modern_answer_chunks.append(chunk)
		if idx % 10 == 0:
			modern_answer_placeholder.write("".join(modern_answer_chunks))
	modern_answer = "".join(modern_answer_chunks)
	modern_answer_placeholder.write(modern_answer)
	gen_status.update(label="Generation completed", state="complete")

	# Step 2: Hamlet Model Output (LoRA merged)
	st.subheader("Hamlet Model Output (LoRA merged)")
	@st.cache_resource
	def load_hamlet_merged_model():
		base_model_name = os.getenv("LORA_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
		adapter_repo = os.getenv("LORA_ADAPTER_REPO", "asulova/hamlet-model")
		base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu")
		tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		tokenizer.pad_token = tokenizer.eos_token
		peft_model = PeftModel.from_pretrained(base_model, adapter_repo)
		merged_model = peft_model.merge_and_unload()
		merged_model.eval()
		return merged_model, tokenizer

	hamlet_model, hamlet_tokenizer = load_hamlet_merged_model()
	hamlet_prompt = build_style_prompt(modern_answer)
	hamlet_status = st.status("Generating with Hamlet (LoRA merged)...", state="running")
	hamlet_placeholder = st.empty()
	hamlet_inputs = hamlet_tokenizer(hamlet_prompt, return_tensors="pt")
	with torch.no_grad():
		hamlet_outputs = hamlet_model.generate(
			**hamlet_inputs,
			max_new_tokens=int(num_predict)
		)
	hamlet_answer = hamlet_tokenizer.decode(hamlet_outputs[0], skip_special_tokens=True)
	hamlet_placeholder.write(hamlet_answer)
	hamlet_status.update(label="Hamlet (LoRA merged) generation completed", state="complete")

	# Step 3: Hamlet-DPO Model Output
	st.subheader("Hamlet-DPO Model Output")
	@st.cache_resource
	def load_hamlet_dpo_model():
		dpo_repo = os.getenv("HAMLET_DPO_MODEL", "asulova/hamlet-dpo")
		model = AutoModelForCausalLM.from_pretrained(dpo_repo, device_map="cpu")
		tokenizer = AutoTokenizer.from_pretrained(dpo_repo)
		tokenizer.pad_token = tokenizer.eos_token
		model.eval()
		return model, tokenizer

	dpo_model, dpo_tokenizer = load_hamlet_dpo_model()
	dpo_prompt = build_style_prompt(modern_answer)
	dpo_status = st.status("Generating with Hamlet-DPO...", state="running")
	dpo_placeholder = st.empty()
	dpo_inputs = dpo_tokenizer(dpo_prompt, return_tensors="pt")
	with torch.no_grad():
		dpo_outputs = dpo_model.generate(
			**dpo_inputs,
			max_new_tokens=int(num_predict)
		)
	dpo_answer = dpo_tokenizer.decode(dpo_outputs[0], skip_special_tokens=True)
	dpo_placeholder.write(dpo_answer)
	dpo_status.update(label="Hamlet-DPO generation completed", state="complete")

	st.subheader("Sources")
	for i, h in enumerate(hits, start=1):
		with st.expander(f"[{i}] {h.source} (score: {h.score:.3f})"):
			st.write(h.text)




