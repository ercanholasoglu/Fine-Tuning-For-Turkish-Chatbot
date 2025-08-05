

import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
login()


df = pd.read_json("hf://datasets/SoAp9035/turkish_instructions/turkish_instructions.json", lines=True)

def format_chat(row):
    return f"### User: {row['user']}\n### Assistant: {row['assistant']}"

df['formatted_text'] = df.apply(format_chat, axis=1)
dataset = Dataset.from_pandas(df[['formatted_text']])
model_name = "meta-llama/Meta-Llama-3-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)


if torch.backends.mps.is_available():
    target_device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS).")
else:
    target_device = torch.device("cpu")
    print("MPS not available, falling back to CPU.")
    print("Warning: Running large models on CPU can be very slow and memory intensive.")


print(f"Attempting to load model to CPU first ({torch.float16})...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    token=HF_TOKEN
)
print(f"Model loaded to CPU. Now moving to {target_device}...")

model.to(target_device)
print(f"Model moved to {target_device}.")


model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

def tokenize_function(examples):
    tokens = tokenizer(
        examples["formatted_text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_special_tokens_mask=True,
    )
    return tokens

dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["formatted_text"]
)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_arguments,
    peft_config=peft_config,

)



trainer.train()
trainer.save_model("./fine_tuned_llama3_8b_for_tr_wqlora")
tokenizer.save_pretrained("./fine_tuned_llama3_8b_for_tr_wqlora")

trainer.push_to_hub("SutskeverFanBoy/fine_tuned_llama3_8b_for_tr_wqlora")
tokenizer.push_to_hub("SutskeverFanBoy/fine_tuned_llama3_8b_for_tr_wqlora")



# Temel modeli ve tokenizer'ı yükleme
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # Llama 3 8B için doğru model adı
fine_tuned_model_path = "./fine_tuned_llama3_8b_for_tr_wqlora"
output_merged_model_path = "./llama3-8b-merged"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
model = PeftModel.from_pretrained(model, fine_tuned_model_path)
model = model.merge_and_unload()
model.save_pretrained(output_merged_model_path)
tokenizer.save_pretrained(output_merged_model_path)

print(f"Llama 3 model başarıyla birleştirildi ve '{output_merged_model_path}' klasörüne kaydedildi.")

!git clone https://github.com/ggerganov/llama.cpp.git
!python3 llama.cpp/convert_hf_to_gguf.py ./llama3-8b-merged/ --outtype q8_0 --outfile ./llama3-8b-merged-q8_0.gguf

from huggingface_hub import HfApi, create_repo


api = HfApi()
repo_id = "SutskeverFanBoy/fine_tuned_llama3_8b_for_tr_gguf"
create_repo(repo_id=repo_id, private=False, exist_ok=True)



gguf_file = "./llama3-8b-merged-q8_0.gguf"
model_folder = "./llama3-8b-merged"


api.upload_file(
    path_or_fileobj=gguf_file,
    path_in_repo=gguf_file.split("/")[-1],
    repo_id=repo_id,
    repo_type="model"
)

print("Diğer model dosyaları yükleniyor...")
api.upload_folder(
    folder_path=model_folder,
    repo_id=repo_id,
    repo_type="model"
)

print("Tüm dosyalar başarıyla Hugging Face'e yüklendi!")

