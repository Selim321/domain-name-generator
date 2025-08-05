import requests
import json
import time
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"  
OUTPUT_FILE = "data/synthetic_domain_dataset.json"

business_descriptions = [
    "organic coffee shop in downtown area",
    "AI-powered resume writing service",
    "luxury vegan skincare for men",
    "marketplace for second-hand tech gadgets",
    "mental health app for teenagers",
    "subscription box for indie board games",
    "eco-friendly cleaning products for homes",
    "app that teaches kids to code with robots",
    "fitness app tailored to elderly people",
    "freelancer platform for artists and designers"
]

def generate_domains(description):
    prompt = (
        f"Suggest 3 brandable domain names for a business described as: \"{description}\".\n"
        "The domain names should be short, creative, and MUST end in .com, .org, or .net.\n"
        "Return only the domain names as a list, one per line, without any other text or numbering."
    )

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code == 200:
        output = response.json()["response"]
        # domains = [
        #     line.strip("- ").strip()
        #     for line in output.split("\n")
        #     if line.strip().startswith("-") or line.strip().endswith((".com", ".org", ".net"))
        # ]
        return output.splitlines()
        # return domains
    else:
        print("Error:", response.text)
        return []

def main():
    dataset = []

    for desc in tqdm(business_descriptions):
        time.sleep(1)  # prevent overloading Ollama
        suggestions = generate_domains(desc)
        dataset.append({
            "business_description": desc,
            "suggested_domains": suggestions
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()