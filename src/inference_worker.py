import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 터미널의 자질구레한 로드 경고문 출력을 차단합니다.
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    if len(sys.argv) < 2:
        print("SAFE")
        return
        
    content = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) >= 3 else "v1.1" # 임시 기본값
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"./model/kanana-safeguard-finetuned-{version}"
    
    # 1. 모델 로드 (점유 시작)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 문장 프롬프트 구성 및 추론
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": ""}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=1, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    gen_idx = input_ids.shape[-1]
    output_token = tokenizer.decode(output_ids[0][gen_idx], skip_special_tokens=True)
    
    # 3. 콘솔에 판별 결과 출력 (이 결과물만 api.py가 읽어감)
    pred_is_unsafe = "UNSAFE" in output_token
    print("UNSAFE" if pred_is_unsafe else "SAFE")
    
    # 4. 프로세스 종료 시 운영체제가 GPU 메모리를 100% 강제 해제합니다.

if __name__ == "__main__":
    main()
