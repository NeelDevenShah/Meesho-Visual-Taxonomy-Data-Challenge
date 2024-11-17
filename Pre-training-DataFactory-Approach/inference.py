# Load data
import pandas as pd
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import torch

# Load the base model (e.g., Qwen2-VL-7B-Instruct)
model_name = "Qwen/Qwen2-VL-7B-Instruct"
cache_dir="/content/new_cache_dir/"
# Load the PEFT adapter
adapter_path = "/content/new_cache_dir/training/checkpoint-100/"
inference_csv_file = "/content/Meesho-Data-Challenge/data/test.csv"

output_file = "test_inf.csv"
fieldnames = ['index', 'prediction']

image_path_template = f"/content/new_cache_dir/aml/test/{image_name}"

# Example usage of processing chunks and writing to a file
chunk_size = 1  # Adjust based on your GPU memory
results_buffer = []

####################################### fun1

print("Loading model")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir=cache_dir  # TODO
)

# Load the PEFT config
peft_config = PeftConfig.from_pretrained(adapter_path) 

# Load the model with the adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Activate the adapter (this step is typically not needed for PEFT models as they're active by default)
model.set_adapter("default")
print("PEFT adapter loaded and set active")

# Load the processor
processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, max_pixels=1280*28*28)
print("Loaded processor")

print("Loading data...")
df = pd.read_csv(inference_csv_file)
print(f"Data loaded. Total rows: {len(df)}")

##############################

# TODO make it dynamic as per the code in dataprep
def process_chunk(chunk):
    messages = []
    for c in chunk.itertuples():
        image_name = c.image_link.split("/")[-1]
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path_template},
                {"type": "text", "text": f"What is the {c.entity_name}?"} #TODO
            ]
        }]
        messages.append(message)

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():  # Disable gradient calculation for inference
        # TODO: Check if the op is concatinated or not
        generated_ids = model.generate(**inputs, max_new_tokens=128)  # Adjust token length based on needs

    # TODO: Check if the useful thing is concatinated or not
    # Trim input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Format the result as a list of dictionaries
    results = [{'index': idx, 'prediction': txt.strip()} for idx, txt in zip(chunk["index"], output_texts)]
    
    return results

def write_to_file(results, output_file, fieldnames):
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(results)

######################################## func2, TODO: check if the buffer thing is removable

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

print("Starting processing...")
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers based on CPU/GPU capacity
    future_to_chunk = {executor.submit(process_chunk, df.iloc[i:i + chunk_size]): i for i in range(0, len(df), chunk_size)}

    for future in as_completed(future_to_chunk):
        try:
            # Check about the buffer storage and also check for the format of storage
            chunk_results = future.result()
            results_buffer.extend(chunk_results)
            print(f"Processed {len(results_buffer)} items so far.")

            # Write to file every 20 processed items
            if len(results_buffer) >= 500:
                write_to_file(results_buffer, output_file, fieldnames)
                print(f"Wrote {len(results_buffer)} items to file.")
                results_buffer = []

        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

# Write any remaining results
if results_buffer:
    write_to_file(results_buffer, output_file, fieldnames)
    print(f"Wrote final {len(results_buffer)} items to file.")

print("Processing completed.")

##################################