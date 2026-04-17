from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import datetime
import uuid
import subprocess

import database

app = FastAPI(title="Risk Detection API")

@app.on_event("startup")
def startup_event():
    database.init_db()
    print("[시스템] SQLite 데이터베이스 연결 및 (통합)초기화 완료")

# =========================================
# 1. 분쟁 리스크 탐지 모델 학습 요청 (POST)
# =========================================
class TrainingRequest(BaseModel):
    dataset_version: str
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
        "python", "train_api_worker.py",
        "--job_id", job_id,
        "--dataset_version", req.dataset_version,
        "--epoch", str(req.epoch),
        "--batch_size", str(req.batch_size),
        "--learning_rate", str(req.learning_rate)
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
async def get_training_job_status(job_id: str):
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="해당 작업을 찾을 수 없습니다.")

    # 공통 반환 스펙
    response = {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "model_name": row["model_name"],
        "version": row["version"]
    }
    
    # 만약 재학습(retrain) 건이라면 상세 내역도 함께 출력 (명세서 외 보너스 기능)
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
    from_date: str
    to_date: str
    include_feedback: bool
    retraining_reason: str

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
            job_id, job_type, base_version, from_date, to_date, 
            include_feedback, retraining_reason, status,
            epoch, batch_size, learning_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, 'retrain', req.base_version, req.from_date, req.to_date,
        req.include_feedback, req.retraining_reason, "queued",
        3, 8, 0.0001 # 사용자 동의 하에 하드코딩된 기본값 사용
    ))
    conn.commit()
    conn.close()

    # 백그라운드 워커는 기존 train_api_worker.py를 그대로 재사용! (dataset_version 값에만 from~to기간 명시)
    cmd = [
        "python", "train_api_worker.py",
        "--job_id", job_id,
        "--dataset_version", f"{req.base_version}-fb-data", 
        "--epoch", "3",
        "--batch_size", "8",
        "--learning_rate", "0.0001"
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
    version: str
    dataset_version: str

@app.post("/ai/evaluations/risk-detection")
async def evaluate_risk_detection(req: EvaluationRequest):
    now = datetime.datetime.now()
    eval_id_suffix = str(uuid.uuid4())[:3].upper()
    eval_id = f"eval_{now.strftime('%Y%m%d')}_{eval_id_suffix}"
    
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO evaluations (evaluation_id, version, dataset_version, status)
        VALUES (?, ?, ?, ?)
    """, (eval_id, req.version, req.dataset_version, "queued"))
    conn.commit()
    conn.close()

    cmd = [
        "python", "eval_api_worker.py",
        "--evaluation_id", eval_id,
        "--version", req.version,
        "--dataset_version", req.dataset_version
    ]
    subprocess.Popen(cmd)

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "evaluation job created",
        "evaluation_id": eval_id,
        "status": "queued"
    }

@app.get("/ai/evaluations/{evaluation_id}")
async def get_evaluation_result(evaluation_id: str):
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations WHERE evaluation_id = ?", (evaluation_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="해당 평가 작업을 찾을 수 없습니다.")

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "version": row["version"],
        "dataset_version": row["dataset_version"],
        "status": row["status"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1_score": row["f1_score"],
        "passed": bool(row["passed"]) if row["passed"] is not None else None
    }

if __name__ == "__main__":
    import uvicorn
    print("\n[알림] 서비스 전용 포트로 서버를 시작합니다.")
    uvicorn.run("api:app", host="0.0.0.0", port=65001, reload=False)
