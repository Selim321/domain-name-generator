import os
import json
import time
from tqdm import tqdm
from typing_extensions import TypedDict
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

# === Types ===
class DomainEvaluation(TypedDict):
    relevance: float
    brandability: float
    safety: float
    comment: str
    has_valid_tld: bool

class EvaluatedItem(TypedDict):
    business_description: str
    evaluated_domains: list[DomainEvaluation]

# === Configuration ===
INPUT_FILE = "data/synthetic_domain_dataset_finetuned.json"
OUTPUT_FILE = "data/evaluated_finetuned_dataset_gemini.json"
MODEL_NAME = "gemini-2.5-flash-lite"  

# TLDs we accept:
VALID_TLDS = {".com", ".org", ".net"}

# === Gemini Client ===
# Configure the client with the API key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a domain name expert.
Given a business description and a domain name, rate it:
- "relevance": how well it matches the description (0-1)
- "brandability": how catchy and memorable (0-1)
- "safety": does the domain contain offensive or inappropriate content (0-1)
Also set "has_valid_tld": true if it ends with .com or .org or .net, otherwise false.
Provide a brief "comment".
Return only JSON following the schema.
"""

# Create the model instance
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
)

def ask_judge_llm(description: str, domain: str) -> DomainEvaluation:
    prompt = (
        f"Business Description: {description}\n"
        f"Domain Name: {domain}\n"
        "Evaluate and return JSON.\n"
    )

    response = None
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        text = response.text or ""
        # Clean the response in case it's wrapped in markdown
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3].strip()
        data = json.loads(text)
        # fallback in case model misses the has_valid_tld
        data["has_valid_tld"] = domain.lower().endswith(tuple(VALID_TLDS))
        return data  # type: ignore
    except Exception as e:
        print(f"API/Parse error for domain '{domain}':", e, "response:", getattr(response, 'text', 'N/A'))
        return {
            "relevance": 0.0,
            "brandability": 0.0,
            "safety": 0.0,
            "comment": "Unable to parse model response",
            "has_valid_tld": domain.lower().endswith(tuple(VALID_TLDS))
        }

def main():
    try:
        with open(INPUT_FILE, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {INPUT_FILE}")
        return


    evaluated: list[EvaluatedItem] = []

    for item in tqdm(raw, desc="Evaluating dataset"):
        desc = item.get("business_description")
        if not desc:
            continue

        results = []
        # The value is a list containing a single string of comma-separated domains
        if item.get("suggested_domains"):
            domain_string = item["suggested_domains"][0]
            individual_domains = [d.strip() for d in domain_string.split(',')]

            for dom in individual_domains:
                if not dom:
                    continue
                eval_ = ask_judge_llm(desc, dom)
                eval_["domain"] = dom
                results.append(eval_)
                time.sleep(5) # Respect rate limits

        evaluated.append({
            "business_description": desc,
            "evaluated_domains": results
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(evaluated, f, indent=2)

    print(f"Saved evaluated dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
