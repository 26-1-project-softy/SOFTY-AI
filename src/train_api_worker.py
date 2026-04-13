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
    args = parser.parse_args()

    conn = database.get_connection()
    cursor = conn.cursor()

    try:
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        cursor.execute("UPDATE training_jobs SET status = 'running', started_at = ? WHERE job_id = ?", (now_str, args.job_id))
        conn.commit()
        print(f"[{args.job_id}] 실제 PyTorch 백그라운드 학습 시작 (Epoch: {args.epoch}, Global Batch: {args.batch_size})")

        # 1. 모델 / 토크나이저 준비
        model_name = "kakaocorp/kanana-safeguard-8b"
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
                if match and int(match.group(1)) >= 31: 
                    param.requires_grad = True

        if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()

        # 3. 데이터 로드 (모든 데이터를 다 학습)
        df = pd.read_csv("./data/dataset-v1.0.csv")
        full_dataset = CustomHFTrainingDataset(df, tokenizer, max_length=512)
        
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
        patience = 1
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
            
            # Epoch 종료 후 Validation 및 저장
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device), 
                        attention_mask=batch["attention_mask"].to(device), 
                        labels=batch["labels"].to(device)
                    )
                    val_loss += outputs.loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model.save_pretrained("./kanana-safeguard-finetuned-v1.0")
                tokenizer.save_pretrained("./kanana-safeguard-finetuned-v1.0")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early Stopping!")
                    break

        # 5. DB 상태 업데이트
        finished_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        created_version = f"v{args.dataset_version[-1] if args.dataset_version[-1].isdigit() else '1.0'}-trained" 
        
        cursor.execute("""
            UPDATE training_jobs SET status = 'completed', finished_at = ?, model_name = 'kanana-risk-detector', version = ? WHERE job_id = ?
        """, (finished_str, created_version, args.job_id))
        conn.commit()
        print(f"[{args.job_id}] 실제 학습 워커 완료 및 DB 반영 성공")

    except Exception as e:
        cursor.execute("UPDATE training_jobs SET status = 'failed' WHERE job_id = ?", (args.job_id,))
        conn.commit()
        print(f"[{args.job_id}] Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
