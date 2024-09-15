import torch
# import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import warnings

# 특정 경고를 무시하도록 설정
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# 기기 설정 (CPU 또는 MPS)
device = torch.device("cpu")

# 데이터 로드
data_path = "./data_finetunned/coffee_finetuning_20240914_witi_total.jsonl"
dataset = Dataset.from_json(data_path)

print("데이터셋 로드 완료")

# 모델 및 토크나이저 로드
BASE_MODEL = "beomi/gemma-ko-2b"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_special_tokens=True)


# Prompt 생성 함수 수정 (instruction과 context 기반으로 생성)
def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['instruction'])):
        prompt_list.append(f"""<bos><start_of_turn>user
{example['instruction'][i]}<end_of_turn>
<start_of_turn>model
{example['response'][i]}<end_of_turn><eos>""")
    return prompt_list

# 데이터셋을 train 데이터로 설정
train_data = dataset

# 첫 번째 데이터의 프롬프트 확인
print(generate_prompt(train_data[:1])[0])

# LoRA 설정
lora_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 모델 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="outputs",
        max_steps=3000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        warmup_steps=0.03,
        learning_rate=2e-4,
        fp16=False,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
        use_mps_device=False  # CPU로 설정
    ),
    peft_config=lora_config,
    formatting_func=generate_prompt,  # 새로운 포맷팅 함수 적용
)

# 훈련 시작
trainer.train()

# 어댑터 모델 저장
ADAPTER_MODEL = "lora_adapter"
trainer.model.save_pretrained(ADAPTER_MODEL)

# 최종 모델 병합 및 저장
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

model = model.merge_and_unload()
model.save_pretrained('./gemma_outputs/gemma-ko-2b-beans-20240915-01')
print("모델 저장 완료")
# 토크나이저를 저장합니다.
tokenizer.save_pretrained('./gemma_outputs/gemma-ko-2b-beans-20240915-01')
print("tokenizer 저장 완료")