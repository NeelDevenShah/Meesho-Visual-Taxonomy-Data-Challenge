import json
import os
import subprocess

# TODO: Paths, verify everything for qwen, as code is for llama

# Define the training arguments
args = dict(
  stage="sft",                        # do supervised fine-tuning
  do_train=True,
  model_name_or_path="Qwen/Qwen2-VL-2B-Instruct", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  dataset="mllm_demo,identity",             # use alpaca and identity datasets
  template="qwen2_vl",                     # use llama3 prompt template
  finetuning_type="lora",                   # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir="qwen2vl_lora",                  # the path to save LoRA adapters
  per_device_train_batch_size=2,               # the batch size
  gradient_accumulation_steps=4,               # the gradient accumulation steps
  lr_scheduler_type="cosine",                 # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  save_steps=1000,                      # save checkpoint every 1000 steps
  learning_rate=5e-5,                     # the learning rate
  num_train_epochs=3.0,                    # the epochs of training
  max_samples=500,                      # use 500 examples in each dataset
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  loraplus_lr_ratio=16.0,                   # use LoRA+ algorithm with lambda=16.0
  fp16=True,                         # use float16 mixed precision training
  use_liger_kernel=True,                   # use liger kernel for efficient training
)

def make_finetuning():
    # Save the arguments as a JSON file
    with open("train_qwen2vl.json", "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)

    # Change the directory to /content/LLaMA-Factory/
    os.chdir("/content/LLaMA-Factory/")

    # Run the training command
    subprocess.run(["llamafactory-cli", "train", "train_qwen2vl.json"])
