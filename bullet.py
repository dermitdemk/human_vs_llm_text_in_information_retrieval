import re
import csv
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- CONFIG ----------------
HF_TOKEN = "HUGGING_FACE_TOKEN"  #Token to login to Hugginface (LLAMA models are gated)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

INPUT_CSV  = "cleaned_tagesschau_1200.csv"
OUTPUT_CSV = "cleaned_tagesschau_with_bullets_1200.csv"

SYSTEM = (
    "Du bist ein extrem präziser Nachrichten-Redakteur.\n"
    "Regeln:\n"
    "1) Schreibe NUR Fakten, die im Text stehen.\n"
    "2) KEINE Vermutungen, KEINE Erklärungen, KEINE Bewertung.\n"
    "3) Wenn etwas unklar ist, schreibe 'UNKLAR'.\n"
    "4) Antworte NUR mit Bulletpoints.\n"
)

BULLET_PROMPT = (
    "Extrahiere die wichtigsten Fakten als Bulletpoints.\n"
    "Jeder Bullet muss aus dem Text belegbar sein.\n"
    "KEINE neuen Details.\n\n"
    "TEXT:\n{chunk}\n"
)

# -------------- Sentence splitting --------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # simple, robust-ish; works well for news prose
    sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]

# -------------- Chunk count rules --------------
def choose_n_parts(n_tokens: int) -> int:
    if n_tokens <= 1200:
        return 1
    if n_tokens <= 1500:
        return 2
    if n_tokens <= 2000:
        return 3
    if n_tokens <= 3000:
        return 4
    return 5

# -------------- Balanced sentence-aware chunking --------------
def chunk_by_sentences_equal_tokens(
    sentences: List[str],
    tokenizer,
    n_parts: int
) -> List[str]:
    if n_parts <= 1 or not sentences:
        return [" ".join(sentences).strip()] if sentences else [""]

    sent_tokens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    total = sum(sent_tokens)
    target = max(1, total // n_parts)

    chunks = []
    cur = []
    cur_tok = 0
    remaining_sents = len(sentences)
    remaining_parts = n_parts

    for i, (s, t) in enumerate(zip(sentences, sent_tokens)):
        remaining_sents = len(sentences) - i
        # if we must ensure at least 1 sentence per remaining part
        must_leave = remaining_parts - 1

        # decide if we should cut before adding this sentence
        if cur and remaining_parts > 1:
            # if adding would overshoot target and we still can leave enough sentences
            if (cur_tok + t > target) and (remaining_sents > must_leave):
                chunks.append(" ".join(cur).strip())
                cur = []
                cur_tok = 0
                remaining_parts -= 1

                # update target for the rest to avoid tiny last chunk
                total_left = sum(sent_tokens[i:])  # includes current sentence now
                target = max(1, total_left // remaining_parts)

        cur.append(s)
        cur_tok += t

    if cur:
        chunks.append(" ".join(cur).strip())

    # If we ended up with fewer chunks than requested (rare), just return what we have.
    # If we ended up with more (also rare), merge extras.
    while len(chunks) > n_parts:
        chunks[-2] = (chunks[-2] + " " + chunks[-1]).strip()
        chunks.pop()

    return chunks

# -------------- HF chat template --------------
def build_chat(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens=500, temperature=0.2, top_p=0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    # remove prompt prefix heuristically
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded.strip()

def clean_bullets(text: str) -> str:
    # keep only bullet-ish lines; fallback to raw if nothing matches
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.startswith(("-", "*", "•"))]
    return "\n".join(bullet_lines) if bullet_lines else "\n".join(lines)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=True)

    #config for rtx 3070
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,   
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=bnb_config,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    # read input csv
    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        if "bullets" not in fieldnames:
            out_fields = fieldnames + ["bullets"]
        else:
            out_fields = fieldnames

        # write output csv (laufend)
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            for row_i, row in enumerate(reader, start=1):
                content = str(row.get("content", "") or "").strip()

                #We fell back to only use artciles with maximum token of 1200, 
                #so chunking was not used in the final analysis
                # token length of full content
                #n_tokens = len(tokenizer.encode(content, add_special_tokens=False))
                #n_parts = choose_n_parts(n_tokens)

                #sents = split_sentences(content)
                #chunks = chunk_by_sentences_equal_tokens(sents, tokenizer, n_parts)

                #all_bullets = []
                #for ci, ch in enumerate(chunks, start=1):
                #    user = BULLET_PROMPT.format(chunk=ch)
                #    chat = build_chat(tokenizer, SYSTEM, user)
                #    out = generate(model, tokenizer, chat, max_new_tokens=500, temperature=0.2, top_p=0.9)
                #    all_bullets.append(clean_bullets(out))

                user = BULLET_PROMPT.format(chunk=content)
                chat = build_chat(tokenizer, SYSTEM, user)
                out = generate(model, tokenizer, chat, max_new_tokens=500, temperature=0.2, top_p=0.9)

                #We fell back to only use artciles with maximum token of 1200, 
                #so chunking was not used in the final analysis
                row["bullets"] = clean_bullets(out)#"\n".join([b for b in clean_bullets(out) if b.strip()]).strip()
                writer.writerow(row)

                if row_i % 10 == 0:
                    print(f"processed {row_i} rows...")

    print("DONE ->", OUTPUT_CSV)

if __name__ == "__main__":

    main()
