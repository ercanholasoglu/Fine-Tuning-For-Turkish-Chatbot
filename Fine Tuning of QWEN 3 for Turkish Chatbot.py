
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
from huggingface_hub import login
from transformers import TrainingArguments

login(token=HF_TOKEN)
dataset_name = "SoAp9035/turkish_instructions"
df = pd.read_json(f"hf://datasets/{dataset_name}/turkish_instructions.json", lines=True)

def format_chat(row):
    return f"<|im_start|>user\n{row['user']}<|im_end|>\n<|im_start|>assistant\n{row['assistant']}<|im_end|>"

df['formatted_text'] = df.apply(format_chat, axis=1)
train_dataset = Dataset.from_pandas(df[['formatted_text']])

model_name = "Qwen/Qwen3-14B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

if not torch.cuda.is_available():
    print("CUDA mevcut değil, lütfen GPU'lu bir ortamda çalıştırmayı deneyin.")
    raise EnvironmentError("Bu işlem için bir NVIDIA GPU'ya ihtiyacınız var.")

print("NVIDIA GPU (CUDA) kullanılıyor.")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=True
)

print("Model yüklendi.")

model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=True,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)


training_arguments = TrainingArguments(
    output_dir="./qwen2_results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)


def formatting_prompts_func(examples):
    return examples["formatted_text"]

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
)

trainer.train()


output_dir = "./fine_tuned_qwen3_13b_for_tr_wqlora"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Fine-tuning tamamlandı ve Qwen3 modeli kaydedildi!")

output_dir = "./fine_tuned_qwen3_13b_for_tr_wqlora"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuning tamamlandı ve model yerel olarak kaydedildi.")

print("Model Hugging Face Hub'a yükleniyor...")
repository_id = "SutskeverFanBoy/fine_tuned_qwen3_13b_for_tr_wqlora"
trainer.push_to_hub(repository_id, token=HF_TOKEN)

ü

base_model_name = "Qwen/Qwen3-14B"
fine_tuned_model_path = "./fine_tuned_qwen3_13b_for_tr_wqlora"

# Base model yükleniyor
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config, 
    device_map="auto"
)


model = PeftModel.from_pretrained(model, fine_tuned_model_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
model.to("cuda")
prompt = "Selam?'"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  
        do_sample=True,      
        temperature=0.7      
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Yanıtı:")
print(response)


base_model_name = "Qwen/Qwen3-14B"
fine_tuned_model_path = "./fine_tuned_qwen3_13b_for_tr_wqlora"
output_merged_model_path = "./qwen3-14b-merged"

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

print(f"Model başarıyla birleştirildi ve '{output_merged_model_path}' klasörüne kaydedildi.")

!ls -laR llama.cpp

!python3 llama.cpp/convert_hf_to_gguf.py ./qwen3-14b-merged/ \
    --outtype q8_0 \
    --outfile ./qwen3-14b-merged-q8_0.gguf

!cp ./qwen3-14b-merged-q8_0.gguf /content/drive/MyDrive/


repo_name = "SutskeverFanBoy/qwen3-14b-merged-q8-quantized-lora"
output_dir = "./qwen3-14b-merged/"


model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)


