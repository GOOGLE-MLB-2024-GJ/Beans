from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 베이스 모델에서 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

# 저장한 모델 경로
model_dir = './gemma_outputs/gemma-2b-it-sum-ko-beans-1'
model = AutoModelForCausalLM.from_pretrained(model_dir)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 모델을 CPU로 이동 (만약 GPU를 쓴다면 'cuda'로 바꿔줘)
model.to("cpu") #cpu

conversation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)


def chat_with_model(input_text):
    # 대화용 프롬프트를 생성
    messages = [{"role": "user", "content": input_text}]

    # 토크나이저로 입력을 프롬프트 형태로 변환
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 모델이 응답을 생성
    # response = conversation_pipeline(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    response = conversation_pipeline(prompt, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, add_special_tokens=True)

    # 모델의 생성된 응답 추출
    generated_text = response[0]["generated_text"]
    model_response = generated_text[len(prompt):]  # 입력 프롬프트를 제거하고 응답만 반환
    return model_response


# 대화를 계속 이어나갈 수 있는 구조
def interactive_chat():
    print("대화형 모드에 오신 것을 환영합니다! '종료'라고 입력하면 대화가 종료됩니다.")
    while True:
        user_input = input("사용자: ")  # 사용자 입력 받기
        if user_input.lower() == "종료":  # '종료'라고 입력하면 대화 종료
            print("대화를 종료합니다.")
            break
        model_reply = chat_with_model(user_input)  # 모델의 응답 받기
        print(f"모델: {model_reply}")  # 모델의 응답 출력

# 대화 시작
interactive_chat()