import json
from pathlib import Path

INPUT_FILE = Path("data/evaluated_dataset_gemini.json")

def summarize_scores(data):
    total_domains = 0
    relevance_sum = 0
    brandability_sum = 0
    safety_sum = 0
    invalid_tld_count = 0

    for item in data:
        for domain_eval in item["evaluated_domains"]:
            relevance_sum += domain_eval["relevance"]
            brandability_sum += domain_eval["brandability"]
            safety_sum += domain_eval["safety"]
            if not domain_eval["has_valid_tld"]:
                invalid_tld_count += 1
            total_domains += 1

    avg_relevance = relevance_sum / total_domains
    avg_brandability = brandability_sum / total_domains
    avg_safety = safety_sum / total_domains
    invalid_tld_pct = (invalid_tld_count / total_domains) * 100

    print("\n📊 Evaluation Summary:")
    print(f"Total domain suggestions evaluated: {total_domains}")
    print(f"Average Relevance:     {avg_relevance:.2f}")
    print(f"Average Brandability:  {avg_brandability:.2f}")
    print(f"Average Safety:        {avg_safety:.2f}")
    print(f"Domains without valid TLD (.com/.org/.net): {invalid_tld_count} ({invalid_tld_pct:.1f}%)")

if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        data = json.load(f)
    summarize_scores(data)