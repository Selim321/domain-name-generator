import requests
import json
import time
from tqdm import tqdm
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2-finetuned:latest"
OUTPUT_FILE = "data/synthetic_domain_dataset_finetuned_with_guardrails.json"

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

def generate_and_validate_domains(description):
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
        # Extract potential domains using a regex to be more robust
        potential_domains = re.findall(r'([a-zA-Z0-9-]+\.(?:com|org|net))', output)
        
        # Validate and clean the domains
        validated_domains = []
        for domain in potential_domains:
            # Remove leading/trailing hyphens and clean up
            cleaned_domain = domain.strip('-. ')
            if cleaned_domain:
                validated_domains.append(cleaned_domain)
        return validated_domains
    else:
        print(f"Error: {response.text}")
        return []

def main():
    dataset = []

    for desc in tqdm(business_descriptions):
        time.sleep(1)  # prevent overloading Ollama
        suggestions = generate_and_validate_domains(desc)
        dataset.append({
            "business_description": desc,
            "suggested_domains": suggestions
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
