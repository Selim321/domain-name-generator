import json
import os

def format_entry(entry: dict) -> dict:
    """Formats a single data entry into the Llama 3 chat template."""
    
    description = entry["business_description"]
    domains = ", ".join(entry["suggested_domains"])
    
    # This structure is based on the Llama 3 instruction format
    return {
        "messages": [
            {"role": "system", "content": "You are a creative assistant that generates domain name ideas based on a business description."},
            {"role": "user", "content": f"Generate 3 domain name suggestions for the following business: {description}"},
            {"role": "assistant", "content": domains}
        ]
    }

def main():
    input_file_path = "/Users/selim/Documents/ML-projects/domain-name-generator/data/finetune_data.json"
    output_file_path = "/Users/selim/Documents/ML-projects/domain-name-generator/data/train_data.jsonl"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    seen_descriptions = set()
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        data = json.load(infile)
        for entry in data:
            if entry["business_description"] not in seen_descriptions:
                formatted_data = format_entry(entry)
                outfile.write(json.dumps(formatted_data) + "\n")
                seen_descriptions.add(entry["business_description"])

    print(f"Processed {len(seen_descriptions)} unique entries.")
    print(f"Formatted data saved to {output_file_path}")

if __name__ == "__main__":
    main()