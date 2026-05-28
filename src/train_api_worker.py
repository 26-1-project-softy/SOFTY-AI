import argparse
import time
import datetime
import database
import pandas as pd
import torch
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import gc
import sqlite3

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

        # 0. 이전 실패 등으로 존재할 수 있는 동일 버전의 불완전한 모델 디렉토리 정리 (새출발)
        target_dir = f"./model/kanana-safeguard-finetuned-{args.target_version}"
        if os.path.exists(target_dir):
            print(f"[{args.job_id}] ⚠️ 이전 실패 이력이 있어 존재하는 불완전한 모델 폴더 '{target_dir}'를 정리하고 새롭게 학습을 진행합니다.", flush=True)
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)

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

        # 3. 전체 데이터 통째로 로딩
        df_raw = pd.read_csv(f"./data/dataset-{args.dataset_version}.csv")
        
        if args.job_type == "retrain":
            print(f"[{args.job_id}] 재학습 모드: 학습 속도 향상을 위해 전체 데이터의 20%만 무작위 추출하고 Hard Negative Mining(OHEM)을 적용합니다.", flush=True)
            df = df_raw.sample(frac=0.2, random_state=42).reset_index(drop=True)
        else:
            # 전체 데이터 100% 셔플
            df = df_raw.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"[{args.job_id}] 데이터 준비 완료: 총 학습 대상 데이터 수 {len(df)}개", flush=True)        
        
        # 문맥 캔버스 길이를 96으로 약간 줄여서 연산 속도 최적화 (대부분의 악플/문장은 짧음)
        full_dataset = CustomHFTrainingDataset(df, tokenizer, max_length=96)
        
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
                # [GPU 자원 관리] 추론/평가 요청 시 자원을 양보하기 위한 일시정지 체크
                LOCK_DIR = ".gpu_locks"
                if not os.path.exists(LOCK_DIR): os.makedirs(LOCK_DIR)
                
                if os.listdir(LOCK_DIR):
                    print(f"[{args.job_id}] ⚠️ 실시간 추론/평가 요청(총 {len(os.listdir(LOCK_DIR))}건)으로 인해 GPU 자원을 양보합니다. (Pause)", flush=True)
                    
                    # 1. 모델과 옵티마이저 상태를 CPU로 이동하여 VRAM 확보
                    model.cpu()
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cpu()
                    
                    # 2. 잔여 메모리 강제 해제
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # DB 상태를 잠시 'paused'로 변경
                    try:
                        temp_conn = sqlite3.connect('jobs.db', timeout=5)
                        temp_conn.execute("UPDATE training_jobs SET status = 'paused' WHERE job_id = ?", (args.job_id,))
                        temp_conn.commit()
                        temp_conn.close()
                    except: pass
                    
                    # 모든 락 파일이 사라질 때까지 대기
                    while os.listdir(LOCK_DIR):
                        time.sleep(1.0)
                        
                    print(f"[{args.job_id}] 락 해제 감지. 이전 프로세스의 VRAM 해제를 위해 5초간 대기합니다...", flush=True)
                    time.sleep(5.0)
                    
                    # 3. 모델과 옵티마이저 상태를 다시 GPU(device)로 복구 (OOM 방어 재시도 로직 적용)
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            print(f"[{args.job_id}] GPU 자원 재점유 시도 중 ({attempt+1}/{max_retries})...", flush=True)
                            model.to(device)
                            for state in optimizer.state.values():
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor):
                                        state[k] = v.to(device)
                            print(f"[{args.job_id}] ✅ GPU 자원 재점유 완료! (Resume)", flush=True)
                            break
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"[{args.job_id}] ⚠️ GPU 재점유 중 OOM 감지. 캐시 정리 후 5초 뒤 재시도합니다: {e}", flush=True)
                                torch.cuda.empty_cache()
                                gc.collect()
                                time.sleep(5.0)
                                if attempt == max_retries - 1:
                                    raise e
                            else:
                                raise e
                    
                    try:
                        temp_conn = sqlite3.connect('jobs.db', timeout=5)
                        temp_conn.execute("UPDATE training_jobs SET status = 'running' WHERE job_id = ?", (args.job_id,))
                        temp_conn.commit()
                        temp_conn.close()
                    except: pass

                outputs = model(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    labels=batch["labels"].to(device)
                )
                
                # [Hard Negative Mining - OHEM]
                # 재학습 시, 손실(Loss)이 0.1 미만인 '너무 쉬운(Easy)' 데이터는 역전파를 생략하여 속도를 대폭 향상
                is_hard = True
                if args.job_type == "retrain" and outputs.loss.item() < 0.1:
                    is_hard = False
                    
                loss = outputs.loss / accumulation
                
                if is_hard:
                    loss.backward()
                
                if (step + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if (step + 1) % 100 == 0:
                    print(f"[Epoch {epoch+1}/{args.epoch}] Step: {step+1}/{len(train_loader)} - Loss: {loss.item() * accumulation:.4f}", flush=True)
                    
                    # Update progress in DB
                    try:
                        total_steps = len(train_loader) * args.epoch
                        current_step = step + 1 + (epoch * len(train_loader))
                        progress_pct = round((current_step / total_steps) * 100.0, 2)
                        
                        import sqlite3
                        db_conn = sqlite3.connect('jobs.db', timeout=5)
                        db_cursor = db_conn.cursor()
                        db_cursor.execute("UPDATE training_jobs SET progress = ? WHERE job_id = ?", (progress_pct, args.job_id))
                        db_conn.commit()
                        db_conn.close()
                    except Exception as e:
                        print(f"[Progress Update Error] {e}")
                
                eval_steps = max(1, len(train_loader) // 4)
                if (step + 1) % eval_steps == 0 or (step + 1) == len(train_loader):
                    model.eval()
                    val_loss = 0
                    print(f"\n[검증] 중간 성능 평가 추론 중... (Step: {step+1})", flush=True)
                    
                    # 검증 전 메모리 파편화 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            v_input_ids = val_batch['input_ids'].to(device)
                            v_labels = val_batch['labels'].to(device)
                            v_outputs = model(v_input_ids, labels=v_labels)
                            val_loss += v_outputs.loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    
                    # 검증 후 메모리 파편화 정리
                    torch.cuda.empty_cache()
                    gc.collect()
                    
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
            UPDATE training_jobs SET status = 'completed', finished_at = ?, model_name = 'kanana-risk-detector', version = ?, progress = 100.0 WHERE job_id = ?
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
