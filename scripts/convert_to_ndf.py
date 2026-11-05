import json
import urllib.request
import re

def process_unified_datasets():
    # Configuration for all datasets to be merged
    datasets_config = [
        {
            "name": "Stanford Alpaca",
            "url": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
            "source": "Stanford Alpaca Dataset by Tatsu Lab (Stanford University)",
            "license": "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
            "q_key": "instruction",
            "a_key": "output",
            "input_key": "input"
        },
        {
            "name": "General Knowledge",
            "url": "https://huggingface.co/datasets/MuskumPillerum/General-Knowledge/resolve/main/output.json?download=true",
            "source": "General-Knowledge Dataset by MuskumPillerum (Hugging Face)",
            "license": "MIT License",
            "q_key": "Question",
            "a_key": "Answer",
            "input_key": None  # This dataset format does not have an 'input' field
        }
    ]

    # Compile regex for forbidden words (case-insensitive)
    forbidden_pattern = re.compile(r'(gpt|qwen|gemma|claude)', re.IGNORECASE)
    output_filename = "NPC-Forge-General-Knowledge-Dataset.jsonl"
    
    total_seen = 0
    skipped_no_input = 0
    skipped_forbidden = 0
    valid_count = 0

    print(f"Starting unified dataset processing. Output will be saved to: {output_filename}")

    # Open file once for line-by-line writing (memory efficient)
    with open(output_filename, "w", encoding="utf-8") as f:
        for ds in datasets_config:
            print(f"\nDownloading {ds['name']} from {ds['url']}...")
            try:
                with urllib.request.urlopen(ds['url']) as response:
                    data = json.load(response)
            except Exception as e:
                print(f"Failed to download {ds['name']}: {e}")
                continue

            print(f"Processing {len(data)} entries from {ds['name']}...")
            
            for item in data:
                total_seen += 1
                question = str(item.get(ds['q_key'], "")).strip()
                answer = str(item.get(ds['a_key'], "")).strip()
                
                # Handle input key only if it exists in the config
                input_text = ""
                if ds['input_key'] and ds['input_key'] in item:
                    input_text = str(item.get(ds['input_key'], "")).strip()

                # RULE 1: Must have no input (empty or whitespace only)
                if input_text != "":
                    skipped_no_input += 1
                    continue
                    
                # RULE 2: No forbidden words in question or answer
                if forbidden_pattern.search(question) or forbidden_pattern.search(answer):
                    skipped_forbidden += 1
                    continue
                    
                # RULE 3: Skip known placeholder outputs
                if "<nooutput>" in answer.lower() or "cannot be fulfilled" in answer.lower():
                    skipped_forbidden += 1
                    continue
                    
                # RULE 4: Clean the question: lowercase and strip trailing spaces, dots, exclamation marks, and question marks
                clean_question = question.lower().strip(' .!?')
                
                # Skip if question is empty after cleaning
                if not clean_question:
                    continue
                    
                # Convert to NPC-Forge standard format
                new_entry = {
                    "input": [clean_question],
                    "output": f"<||completion||>\n\n{answer}",
                    "permission": "ask",
                    "tools": [], 
                    "source": ds['source'],
                    "license": ds['license']
                }
                
                # Write as a single JSON line
                f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                valid_count += 1
                
    print("\n" + "="*40)
    print("UNIFIED CONVERSION COMPLETE")
    print("="*40)
    print(f"Total entries seen across all datasets: {total_seen}")
    print(f"Skipped (had input text): {skipped_no_input}")
    print(f"Skipped (forbidden words/placeholders/empty): {skipped_forbidden}")
    print(f"Valid entries saved: {valid_count}")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    process_unified_datasets()