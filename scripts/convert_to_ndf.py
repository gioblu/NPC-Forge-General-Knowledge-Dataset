import json
import urllib.request
import re
from pathlib import Path

def process_item(question, answer, input_text, ds, forbidden_pattern, generative_pattern, f, counters):
    # 1. CLEAN STRINGS: Strip whitespace, then strip leading/trailing quotes, then strip whitespace again
    question = str(question).strip().strip('"\'').strip()
    answer = str(answer).strip().strip('"\'').strip()
    input_text = str(input_text).strip().strip('"\'').strip()

    # RULE 0: SKIP GENERATIVE / CREATIVE TASKS (Strict Factual QA only)
    if generative_pattern.search(question):
        counters['skipped_generative'] += 1
        return

    # RULE 1: Must have no input/context (empty or whitespace only)
    if input_text != "":
        counters['skipped_no_input'] += 1
        return
        
    # RULE 2: No forbidden words in question or answer (e.g., specific LLM names)
    if forbidden_pattern.search(question) or forbidden_pattern.search(answer):
        counters['skipped_forbidden'] += 1
        return
        
    # RULE 3: Skip known placeholder outputs
    if "<nooutput>" in answer.lower() or "cannot be fulfilled" in answer.lower():
        counters['skipped_forbidden'] += 1
        return
        
    # RULE 4: Clean the question: lowercase and strip trailing spaces, dots, exclamation marks, question marks, and quotes
    clean_question = question.lower().strip(' .!?"\'')
    
    # Skip if question is empty after cleaning
    if not clean_question:
        counters['skipped_forbidden'] += 1
        return
        
    # Convert to NPC-Forge standard format
    new_entry = {
        "input": [clean_question],
        "output": [f"<||completion||>\n\n{answer}"],
        "permission": "yolo",
        "source": ds['source'],
        "license": ds['license']
    }
    
    # Write as a single JSON line
    f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
    counters['valid_count'] += 1

def process_unified_datasets():
    # Configuration for all datasets to be merged
    datasets_config = [
        {
            "name": "General Knowledge (Hugging Face)",
            "url": "https://huggingface.co/datasets/MuskumPillerum/General-Knowledge/resolve/main/output.json?download=true",
            "source": "General-Knowledge Dataset by MuskumPillerum",
            "license": "MIT License",
            "type": "standard",
            "q_key": "Question",
            "a_key": "Answer",
            "input_key": None
        },
        {
            "name": "Jeopardy Facts (History, Science, Geography)",
            "url": "https://raw.githubusercontent.com/alexhanna/jeopardy/master/jeopardy.json",
            "source": "Jeopardy Questions Dataset (Alex Hanna)",
            "license": "Public Domain / CC0",
            "type": "standard",
            "q_key": "Question",
            "a_key": "Answer",
            "input_key": None
        },
        {
            "name": "COMQA (Complex Factoid QA)",
            "url": "https://qa.mpi-inf.mpg.de/comqa/comqa_train.json",
            "source": "COMQA Dataset by MPI-INF",
            "license": "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
            "type": "comqa",
            "input_key": None
        }
    ]

    # Compile regex for forbidden words (case-insensitive)
    forbidden_pattern = re.compile(r'(gpt|qwen|gemma|claude|llama)', re.IGNORECASE)
    
    # Compile regex to SKIP generative/creative tasks (Strict Factual QA only)
    generative_pattern = re.compile(r'^(write|summarize|translate|generate|create|compose|draft|list \d+|give me a list|explain how to build|code a|program a)', re.IGNORECASE)
    
    output_filename = "dataset_NPC-Forge-Strict-Factual-QA-Dataset.jsonl"
    
    counters = {
        'total_seen': 0,
        'skipped_no_input': 0,
        'skipped_forbidden': 0,
        'skipped_generative': 0,
        'valid_count': 0
    }

    print(f"Starting unified factual QA dataset processing. Output: {output_filename}")

    with open(output_filename, "w", encoding="utf-8") as f:
        for ds in datasets_config:
            data = []
            
            # 1. Fetch Data
            if ds.get("type") in ["local", "local_dir"]:
                path_obj = Path(ds["path"]).resolve()
                print(f"\nProcessing local path: {path_obj}")
                
                if path_obj.is_file():
                    # It's a specific file
                    try:
                        with open(path_obj, "r", encoding="utf-8") as file:
                            file_data = json.load(file)
                            if isinstance(file_data, list):
                                data.extend(file_data)
                            else:
                                data.append(file_data)
                        print(f"  -> Loaded {len(data)} entries from {path_obj.name}")
                    except Exception as e:
                        print(f"  -> [ERROR] Failed to load {path_obj.name}: {e}")
                        
                elif path_obj.is_dir():
                    # It's a directory
                    json_files = sorted(path_obj.glob("*.json"))
                    if not json_files:
                        print(f"  -> [WARNING] No .json files found in {path_obj}. Skipping.")
                        continue
                        
                    for json_file in json_files:
                        try:
                            with open(json_file, "r", encoding="utf-8") as file:
                                file_data = json.load(file)
                                if isinstance(file_data, list):
                                    data.extend(file_data)
                                else:
                                    data.append(file_data)
                        except Exception as e:
                            print(f"  -> [ERROR] Failed to load {json_file.name}: {e}")
                else:
                    print(f"  -> [WARNING] Path {path_obj} does not exist. Skipping.")
                    continue
            else:
                print(f"\nDownloading {ds['name']} from {ds.get('url', 'N/A')}...")
                try:
                    req = urllib.request.Request(ds['url'], headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as response:
                        text = response.read().decode('utf-8')
                        if ds['url'].endswith('.jsonl'):
                            data = [json.loads(line) for line in text.strip().split('\n') if line.strip()]
                        else:
                            data = json.loads(text)
                except Exception as e:
                    print(f"Failed to download {ds['name']}: {e}")
                    continue

            print(f"Processing {len(data)} entries from {ds['name']}...")
            
            # 2. Process Data based on type
            if ds.get("type") == "comqa":
                for item in data:
                    counters['total_seen'] += 1
                    questions = item.get("questions", [])
                    answers = item.get("answers", [])
                    if not questions or not answers:
                        continue
                    
                    answer_text = ", ".join(str(a) for a in answers)
                    for q in questions:
                        process_item(q, answer_text, "", ds, forbidden_pattern, generative_pattern, f, counters)
                        
            elif ds.get("type") in ["local", "local_dir"]:
                q_keys = ["question", "Question", "instruction", "prompt", "query", "text", "clue"]
                a_keys = ["answer", "Answer", "output", "response", "completion", "value"]
                
                for item in data:
                    counters['total_seen'] += 1
                    if not isinstance(item, dict):
                        continue
                        
                    question = ""
                    for k in q_keys:
                        if k in item:
                            question = str(item[k])
                            break
                            
                    answer = ""
                    for k in a_keys:
                        if k in item:
                            answer = str(item[k])
                            break
                            
                    input_text = ""
                    if "context" in item:
                        input_text = str(item["context"])
                    elif "input_text" in item:
                        input_text = str(item["input_text"])
                        
                    process_item(question, answer, input_text, ds, forbidden_pattern, generative_pattern, f, counters)
                    
            else: # standard (General Knowledge, Jeopardy)
                for item in data:
                    counters['total_seen'] += 1
                    question = str(item.get(ds.get("q_key", ""), ""))
                    answer = str(item.get(ds.get("a_key", ""), ""))
                    
                    input_text = ""
                    if ds.get("input_key") and ds["input_key"] in item:
                        input_text = str(item.get(ds["input_key"], ""))
                    
                    process_item(question, answer, input_text, ds, forbidden_pattern, generative_pattern, f, counters)
                    
    print("\n" + "="*60)
    print("UNIFIED FACTUAL QA CONVERSION COMPLETE")
    print("="*60)
    print(f"Total entries seen across all datasets: {counters['total_seen']}")
    print(f"Skipped (had input/context text): {counters['skipped_no_input']}")
    print(f"Skipped (generative/creative tasks): {counters['skipped_generative']}")
    print(f"Skipped (forbidden words/placeholders): {counters['skipped_forbidden']}")
    print(f"Valid factual QA entries saved: {counters['valid_count']}")
    print(f"Output saved to: {output_filename}")

if __name__ == "__main__":
    process_unified_datasets()