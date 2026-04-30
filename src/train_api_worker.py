import argparse
import time
import datetime
import database
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class CustomHFTrainingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        label_mapping = {0: "<UNSAFE>", 1: "<SAFE>"}
        self.df['target_text'] = self.df['label'].map(label_mapping).fillna(self.df['label'].astype(str))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        user_text = str(self.df['content'].iloc[idx])
        target_text = str(self.df['target_text'].iloc[idx])
        messages = [{"role": "user", "content": user_text}, {"role": "assistant", "content": target_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    parser = argparse.ArgumentParser(description="API Background Training Worker")
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--dataset_version", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--job_type", type=str, default="train")
    parser.add_argument("--base_version", type=str, default="")
    parser.add_argument("--target_version", type=str, default="v1.1")
    args = parser.parse_args()

    conn = database.get_connection()
    cursor = conn.cursor()

    try:
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        cursor.execute("UPDATE training_jobs SET status = 'running', started_at = ? WHERE job_id = ?", (now_str, args.job_id))
        conn.commit()
        print(f"[{args.job_id}] 실제 PyTorch 백그라운드 학습 시작 (Epoch: {args.epoch}, Batch: {args.batch_size}, Type: {args.job_type})", flush=True)

        # 1. 모델 / 토크나이저 준비
        if args.job_type == "retrain":
            model_name = f"./model/kanana-safeguard-finetuned-{args.base_version}"
            print(f"[{args.job_id}] 재학습(Retrain) 모드: {model_name} 에서 과거 지능(가중치)을 불러옵니다.", flush=True)
        else:
            model_name = "kakaocorp/kanana-safeguard-8b"
            print(f"[{args.job_id}] 초기 학습(Train) 모드: 카카오 순정 베이스 모델을 새로 다운받아 백지에서 훈련합니다.", flush=True)
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

        # 2. VRAM 절약을 위한 마지막 레이어 Freezing 세팅
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lm_head" in name or "norm" in name:
                param.requires_grad = True
            else:
                match = re.search(r'layers?\.([0-9]+)\.', name)
                if match and int(match.group(1)) >= 30: # Layer 30, 31 (총 2개층) 개방 (OOM 방지)
                    param.requires_grad = True

        if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()

        # 3. 전체 데이터 통째로 로딩 및 약점(Recall) 극복을 위한 오버샘플링
        df_raw = pd.read_csv(f"./data/dataset-{args.dataset_version}.csv")
        
        # UNSAFE(0) 데이터만 1.5배로 복제하여 모델이 치명적 위험을 더 자주 보게 함
        df_unsafe = df_raw[df_raw['label'] == 0]
        df_safe = df_raw[df_raw['label'] == 1]
        df = pd.concat([df_unsafe, df_unsafe.sample(frac=0.5, random_state=42), df_safe]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[{args.job_id}] 데이터 오버샘플링 완료: 총 데이터 수 {len(df)}개 (UNSAFE 비중 확대 적용)", flush=True)        
        # 문맥 캔버스 길이를 128로 유지하여 어텐션(N^2) 연산 속도 16x 최적화 방어
        full_dataset = CustomHFTrainingDataset(df, tokenizer, max_length=128)
        
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        # API 요청으로 들어온 batch_size는 OOM 방지를 위해 실제로는 무조건 batch=1로 돌리되 gradient_accumulation으로 맞춰줍니다.
        actual_batch = 1
        accumulation = args.batch_size 
        
        train_loader = DataLoader(train_dataset, batch_size=actual_batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=actual_batch, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate) 
        
        best_val_loss = float('inf')
        patience = 3 # 섣부른 조기 종료 방지
        patience_counter = 0

        # 4. 학습 루프 수행
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                outputs = model(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    labels=batch["labels"].to(device)
                )
                loss = outputs.loss / accumulation
                loss.backward()
                
                if (step + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if (step + 1) % 100 == 0:
                    print(f"[Epoch {epoch+1}/{args.epoch}] Step: {step+1}/{len(train_loader)} - Loss: {loss.item() * accumulation:.4f}", flush=True)
                
                eval_steps = max(1, len(train_loader) // 4)
                if (step + 1) % eval_steps == 0 or (step + 1) == len(train_loader):
                    model.eval()
                    val_loss = 0
                    print(f"[검증] 중간 성능 평가 추론 중... (Step: {step+1})", flush=True)
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_outputs = model(
                                input_ids=val_batch["input_ids"].to(device), 
                                attention_mask=val_batch["attention_mask"].to(device), 
                                labels=val_batch["labels"].to(device)
                            )
                            val_loss += val_outputs.loss.item()
                    avg_val_loss = val_loss / len(val_loader)
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        output_dir = f"./model/kanana-safeguard-finetuned-{args.target_version}"
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        print(f"👉 [체크포인트 성능 갱신] Loss: {avg_val_loss:.4f} -> 폴더 점유율 확보 및 베스트 모델 저장 완료!", flush=True)
                    else:
                        patience_counter += 1
                        print(f"👉 [성능 미갱신] Loss: {avg_val_loss:.4f} -> Patience: {patience_counter}/{patience}", flush=True)
                        if patience_counter >= patience:
                            print("🚨 과적합 징후 포착! 조기 종료(Early Stopping) 발동!", flush=True)
                            break
                    model.train() # 학습 모드 복귀
                    
            if patience_counter >= patience:
                break

        # 5. DB 상태 업데이트
        finished_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        created_version = args.target_version
        
        cursor.execute("""
            UPDATE training_jobs SET status = 'completed', finished_at = ?, model_name = 'kanana-risk-detector', version = ? WHERE job_id = ?
        """, (finished_str, created_version, args.job_id))
        conn.commit()
        print(f"[{args.job_id}] 실제 학습 워커 완료 및 DB 반영 성공", flush=True)

    except Exception as e:
        cursor.execute("UPDATE training_jobs SET status = 'failed' WHERE job_id = ?", (args.job_id,))
        conn.commit()
        print(f"[{args.job_id}] Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
