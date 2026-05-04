from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import datetime
import uuid
import subprocess
import httpx
import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()  # .env 파일 로드

import database

app = FastAPI(title="Risk Detection API")

@app.on_event("startup")
def startup_event():
    database.init_db()
    print("[시스템] SQLite 데이터베이스 연결 및 (통합)초기화 완료")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Kanana API Server is running!"}

# =========================================
# 1. 분쟁 리스크 탐지 모델 학습 요청 (POST)
# =========================================
class TrainingRequest(BaseModel):
    dataset_version: str
    target_version: str
    epoch: int
    batch_size: int
    learning_rate: float

@app.post("/ai/training-jobs/risk-detection")
async def request_training_job(req: TrainingRequest):
    now = datetime.datetime.now()
    job_id_suffix = str(uuid.uuid4())[:3].upper()
    job_id = f"train_{now.strftime('%Y%m%d')}_{job_id_suffix}"
    
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_jobs (job_id, job_type, dataset_version, epoch, batch_size, learning_rate, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (job_id, 'train', req.dataset_version, req.epoch, req.batch_size, req.learning_rate, "queued"))
    conn.commit()
    conn.close()

    cmd = [
        "python", "src/train_api_worker.py",
        "--job_id", job_id,
        "--dataset_version", req.dataset_version,
        "--epoch", str(req.epoch),
        "--batch_size", str(req.batch_size),
        "--learning_rate", str(req.learning_rate),
        "--job_type", "train",
        "--target_version", req.target_version
    ]
    subprocess.Popen(cmd)

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "training job created",
        "job_id": job_id,
        "status": "queued"
    }

# =========================================
# 2. 통합 학습 상태 조회 요청 (GET)
# =========================================
@app.get("/ai/training-jobs/{job_id}")
@app.get("/ai/training-jobs")
async def get_training_job_status(job_id: Optional[str] = None):
    conn = database.get_connection()
    cursor = conn.cursor()
    
    if job_id:
        cursor.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
    else:
        # ID가 없는 경우 가장 최근 '성공한(completed)' 학습 이력 1건 조회
        cursor.execute("SELECT * FROM training_jobs WHERE status = 'completed' ORDER BY ROWID DESC LIMIT 1")
        
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="해당 작업을 찾을 수 없습니다.")

    # 공통 반환 스펙
    response = {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "job_id": row["job_id"],
        "dataset_version": row["dataset_version"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "model_name": row["model_name"],
        "version": row["version"]
    }
    
    # 만약 재학습(retrain) 건이라면 상세 내역도 함께 출력
    if row["job_type"] == "retrain":
        response["base_version"] = row["base_version"]
        response["from_date"] = row["from_date"]
        response["to_date"] = row["to_date"]
        
    return response

# =========================================
# 3. 운영 결과 기반 재학습 요청 (POST)
# =========================================
class RetrainingRequest(BaseModel):
    base_version: str
    target_version: str
    from_date: str
    to_date: str
    include_feedback: bool
    retraining_reason: str
    epoch: int = 3
    batch_size: int = 8
    learning_rate: float = 0.0001

@app.post("/ai/retraining-jobs/risk-detection")
async def request_retraining_job(req: RetrainingRequest):
    now = datetime.datetime.now()
    job_id_suffix = str(uuid.uuid4())[:3].upper()
    job_id = f"retrain_{now.strftime('%Y%m%d')}_{job_id_suffix}"
    
    conn = database.get_connection()
    cursor = conn.cursor()
    # 확장된 컬럼에 재학습 정보 삽입
    cursor.execute("""
        INSERT INTO training_jobs (
            job_id, job_type, dataset_version, base_version, from_date, to_date, 
            include_feedback, retraining_reason, status,
            epoch, batch_size, learning_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, 'retrain', f"{req.base_version}-fb-data", req.base_version, req.from_date, req.to_date,
        req.include_feedback, req.retraining_reason, "queued",
        req.epoch, req.batch_size, req.learning_rate
    ))
    conn.commit()
    conn.close()

    # 백그라운드 워커는 기존 train_api_worker.py를 그대로 재사용! (dataset_version 값에만 from~to기간 명시)
    cmd = [
        "python", "src/train_api_worker.py",
        "--job_id", job_id,
        "--dataset_version", f"{req.base_version}-fb-data", 
        "--epoch", str(req.epoch),
        "--batch_size", str(req.batch_size),
        "--learning_rate", str(req.learning_rate),
        "--job_type", "retrain",
        "--base_version", req.base_version,
        "--target_version", req.target_version
    ]
    subprocess.Popen(cmd)

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "retraining job created",
        "job_id": job_id,
        "status": "queued"
    }

# =========================================
# 4. 및 5. 성능 평가(Evaluation) 파트
# =========================================
class EvaluationRequest(BaseModel):
    version: Optional[str] = None
    dataset_version: Optional[str] = None

@app.post("/ai/evaluations/risk-detection")
async def evaluate_risk_detection(req: EvaluationRequest):
    now = datetime.datetime.now()
    eval_id_suffix = str(uuid.uuid4())[:3].upper()
    eval_id = f"eval_{now.strftime('%Y%m%d')}_{eval_id_suffix}"
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # version이나 dataset_version이 없을 경우 가장 최근의 성공한 평가 파라미터 사용
    reference_eval_id = None
    if not req.version or not req.dataset_version:
        cursor.execute("SELECT evaluation_id, version, dataset_version FROM evaluations WHERE status = 'completed' AND passed = 1 ORDER BY ROWID DESC LIMIT 1")
        last_success = cursor.fetchone()
        if not last_success:
            conn.close()
            raise HTTPException(status_code=400, detail="이전 성공 이력이 없어 version과 dataset_version을 생략할 수 없습니다.")
            
        final_version = req.version if req.version else last_success["version"]
        final_dataset_version = req.dataset_version if req.dataset_version else last_success["dataset_version"]
        reference_eval_id = last_success["evaluation_id"]
    else:
        final_version = req.version
        final_dataset_version = req.dataset_version

    cursor.execute("""
        INSERT INTO evaluations (evaluation_id, version, dataset_version, status)
        VALUES (?, ?, ?, ?)
    """, (eval_id, final_version, final_dataset_version, "queued"))
    conn.commit()
    conn.close()

    cmd = [
        "python", "src/eval_api_worker.py",
        "--evaluation_id", eval_id,
        "--version", final_version,
        "--dataset_version", final_dataset_version
    ]
    subprocess.Popen(cmd)

    response = {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "evaluation job created",
        "evaluation_id": eval_id,
        "status": "queued"
    }
    
    if reference_eval_id:
        response["reference_evaluation_id"] = reference_eval_id
        
    return response

@app.get("/ai/evaluations/{evaluation_id}")
@app.get("/ai/evaluations")
async def get_evaluation_result(evaluation_id: Optional[str] = None):
    conn = database.get_connection()
    cursor = conn.cursor()
    
    if evaluation_id:
        cursor.execute("SELECT * FROM evaluations WHERE evaluation_id = ?", (evaluation_id,))
    else:
        # ID가 없는 경우 가장 최근 평가 이력 1건 조회 (ROWID를 사용하여 삽입 순서대로 정렬)
        cursor.execute("SELECT * FROM evaluations ORDER BY ROWID DESC LIMIT 1")
        
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="해당 평가 작업을 찾을 수 없습니다.")

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "evaluation_id": row["evaluation_id"],
        "version": row["version"],
        "dataset_version": row["dataset_version"],
        "status": row["status"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1_score"],
        "passed": bool(row["passed"]) if row["passed"] is not None else None
    }

# =========================================
# 6. 실시간 추론 (Inference) API 요청 (POST)
# =========================================
class InferenceRequest(BaseModel):
    content: str
    version: str = "v1.1"

@app.post("/ai/inference/risk-detection")
async def infer_risk_detection(req: InferenceRequest):
    import subprocess
    
    print("[추론 API] 추론 요청 수신. 공용 서버 정책에 따라 단발성 스크립트를 띄웁니다...")
    
    # inference_worker.py를 subprocess로 실행하여 모델을 켜고, 끝나면 OS가 즉시 메모리를 해제함.
    proc = subprocess.run(
        ["python", "src/inference_worker.py", req.content, req.version], 
        capture_output=True, 
        text=True
    )
    
    # 출력된 문자열(stdout)의 마지막 줄에서 결과(SAFE/UNSAFE) 추출
    out_text = proc.stdout.strip()
    prediction = "UNSAFE" if "UNSAFE" in out_text else "SAFE"
    
    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "prediction": prediction
    }

# =========================================
# 7. 외부 거대언어모델(LLM) API 연동 추론
# =========================================

EXTERNAL_API_URL = os.getenv("EXTERNAL_API_URL", "http://cellm.gachon.ac.kr:8080/v1/chat/completions")
TEAM_KEY = os.getenv("TEAM_KEY")

class RecommendRequest(BaseModel):
    content: str

class ClassifyRequest(BaseModel):
    content: str

@app.post("/ai/inference/recommend-alternative")
async def recommend_alternative(req: RecommendRequest):
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "text",
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 친절한 언어 교정가입니다. 주어진 문장의 핵심 의미는 유지하되, 공격적이거나 유해한 표현을 완전히 순화하여 예의 바르고 긍정적인 형태의 문장으로 1문장만 재작성하여 대답하세요."
                },
                {"role": "user", "content": req.content}
            ],
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {TEAM_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = await client.post(EXTERNAL_API_URL, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
        except Exception as e:
            print(f"External API Error: {e}")
            raise HTTPException(status_code=500, detail="외부 API 통신에 실패했습니다.")
        
        result = response.json()
        recommended_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        return {
            "content_type": "json",
            "result_code": 200,
            "result_msg": "success",
            "recommended_sentence": recommended_text
        }

@app.post("/ai/inference/classify-intent")
async def classify_intent(req: ClassifyRequest):
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "text",
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 문장 의도 분류기입니다. 주어진 문장의 의도를 파악하여 반드시 다음 5가지 단어 중 하나로만 대답하세요: [출결, 상담, 요청, 문의, 기타]. 특히 지각, 결석, 조퇴, 출석체크 등 학교 수업 참여와 관련된 내용은 질문이나 요청 형태라도 반드시 '출결'로 분류하세요. 다른 부가 설명은 절대 작성하지 마세요."
                },
                {"role": "user", "content": req.content}
            ],
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {TEAM_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = await client.post(EXTERNAL_API_URL, json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()
        except Exception as e:
            print(f"External API Error: {e}")
            raise HTTPException(status_code=500, detail="외부 API 통신에 실패했습니다.")
        
        result = response.json()
        intent = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        # 좀 더 확실하게 5가지 키워드 중 하나만 파싱되도록 방어 로직 추가
        valid_intents = ["출결", "상담", "요청", "문의", "기타"]
        final_intent = "기타"
        for v in valid_intents:
            if v in intent:
                final_intent = v
                break
        
        return {
            "content_type": "json",
            "result_code": 200,
            "result_msg": "success",
            "intent": final_intent
        }

if __name__ == "__main__":
    import uvicorn
    print("\n[알림] 서비스 전용 포트로 서버를 시작합니다.")
    uvicorn.run("api:app", host="0.0.0.0", port=65001, reload=False)
