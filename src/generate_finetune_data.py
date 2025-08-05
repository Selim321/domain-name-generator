import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_data():
    """Generates a dataset of domain name descriptions and domain names."""
    prompt = """
    Generate a dataset of 100 business descriptions. For each description, provide 3 relevant and creative domain name suggestions.
    The output should be a single JSON array of objects.
    Each object in the list should have two keys: "business_description" and "suggested_domains".
    "business_description" should be a string describing a website idea.
    "suggested_domains" should be a list of 3 strings, where each string is a potential domain name.

    Example:
    [
      {
        "business_description": "A platform for artists to showcase and sell their digital artwork.",
        "suggested_domains": ["artify.io", "pixelgallery.com", "digi-art.store"]
      }
    ]
    """
    response = model.generate_content(prompt)
    try:
        # Clean the response by removing markdown formatting
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding JSON: {e}")
        raw_response_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_finetune_response.txt")
        print(f"Saving raw response to {raw_response_path}")
        with open(raw_response_path, "w") as f:
            f.write(response.text)
        return None

def main():
    """Main function to generate and save the dataset."""
    print("Generating data with Gemini...")
    generated_data = generate_data()

    if generated_data is None:
        print("Failed to generate data. Exiting.")
        return

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "finetune_data.json")
    print(f"Saving dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(generated_data, f, indent=2)

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
