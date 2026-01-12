import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------- CONFIG ----------------
HF_TOKEN = "HUGGING_FACE_TOKEN"  #Token to login to Hugginface (LLAMA models are gated)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV  = "cleaned_tagesschau_with_bullets_1200.csv"
OUTPUT_CSV = "cleaned_tagesschau_llm_text_1200.csv"

MIN_NEW_TOKENS = 500
MAX_NEW_TOKENS = 1200

SYSTEM = (
    "Du bist ein extrem präziser Nachrichten-Redakteur.\n"
    "Regeln:\n"
    "1) Schreibe NUR darüber, was in den Bullets steht.\n"
    "2) KEINE Vermutungen, KEINE Erklärungen, KEINE Bewertung.\n"
    "3) Schreibe einen vollständigen Artikel als Fließtext mit Absätzen.\n"
    "4) KEINE neuen Fakten (keine zusätzlichen Zahlen, Zitate, Daten, Namen).\n"
    f"5) Länge: mindestens {MIN_NEW_TOKENS} Tokens, höchstens {MAX_NEW_TOKENS} Tokens.\n"
)

ARTICLE_PROMPT = (
    "Schreibe einen vollständigen Nachrichtenartikel basierend NUR auf den Bullets.\n"
    f"Länge: mindestens {MIN_NEW_TOKENS} Tokens, höchstens {MAX_NEW_TOKENS} Tokens.\n"
    "Alle Aussagen müssen durch Bullets belegbar sein.\n"
    "KEINE neuen Details.\n\n"
    "Bullets:\n{bullets}\n"
)

# -------------- Hugging Face chat template --------------
def build_chat(tokenizer, system: str, user: str) -> str:
    """
    Construct a chat prompt for a Hugging Face LLM using a system prompt and user input.

    Args:
    - tokenizer: A tokenizer object
    - system (str): The system prompt
    - user (str): The user input

    Returns:
    - str: A formatted chat prompt string ready for model input
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def clean_bullets(text: str) -> str:
    """
    Extract bullet points from the given text.

    Args:
    - text (str): The input text

    Returns:
    - str: A string containing lines that start with a bullet marker
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if ln.startswith(("-", "*", "•"))]
    return "\n".join(bullet_lines) if bullet_lines else "\n".join(lines)

@torch.no_grad()
def generate_one_pass(
    model, tokenizer, prompt: str,
    min_new_tokens: int,
    max_new_tokens: int,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """
    Generate text in one pass with a Hugging Face LLM.

    Args:
    - model: A Hugging Face model
    - tokenizer: Corresponding tokenizer for the model
    - prompt (str): Input prompt
    - min_new_tokens (int): Minimum number of new tokens to generate
    - max_new_tokens (int): Maximum number of new tokens to generate
    - temperature (float): Sampling temperature (default is 0.2)
    - top_p (float): Probability threshold for token selection (default is 0.9)

    Returns:
    - str: The generated text with the original prompt removed
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Prefer min_new_tokens if supported; otherwise fallback to min_length
    try:
        gen_kwargs["min_new_tokens"] = min_new_tokens
        out = model.generate(**gen_kwargs)
    except TypeError:
        # Older transformers: enforce minimum via min_length = prompt_len + min_new_tokens
        gen_kwargs["min_length"] = input_len + min_new_tokens
        out = model.generate(**gen_kwargs)

    # Decode ONLY newly generated tokens
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    """
    Generate news articles from the bullet-point summaries in a CSV using a Hugging Face LLM.

    Process:
        1. Load the tokenizer and the model 
        2. Read the input CSV specified by INPUT_CSV
        3. For each row clean the bullets, build a chat prompt, generate text and add the 
           generated text to the new column "llm_text" in the output CSV
        4. Write results in OUTPUT_CSV

    Returns:
    - None.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=True)

    #for rtx 3070
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

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []

        out_col = "llm_text"
        out_fields = fieldnames + ([out_col] if out_col not in fieldnames else [])

        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            for row_i, row in enumerate(reader, start=1):

                # resume logic, in case of script pause/resume
                #idx = row.get("index", "")
                #try:
                #    if idx != "" and int(idx) <= 2960:
                #        continue
                #except ValueError:
                #    pass

                bullets_raw = str(row.get("bullets", "") or "").strip()
                bullets = clean_bullets(bullets_raw)

                if not bullets:
                    row[out_col] = ""
                    writer.writerow(row)
                    continue

                user = ARTICLE_PROMPT.format(bullets=bullets)
                chat = build_chat(tokenizer, SYSTEM, user)

                article = generate_one_pass(
                    model, tokenizer, chat,
                    min_new_tokens=MIN_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.2,
                    top_p=0.9,
                )

                row[out_col] = article
                writer.writerow(row)

                if row_i % 10 == 0:
                    print(f"processed {row_i} rows...")

    print("DONE ->", OUTPUT_CSV)

if __name__ == "__main__":
    main()

