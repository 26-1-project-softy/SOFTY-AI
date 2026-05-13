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
        "progress_percent": row.get("progress", 0.0) if row["status"] != "completed" else 100.0,
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

@app.get("/ai/training-history")
async def get_training_history(page: int = 1, page_size: int = 20):
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # 1. 전체 개수 조회
    query_count = """
        SELECT COUNT(*) as total FROM (
            SELECT t.version FROM training_jobs t WHERE t.version IS NOT NULL
            UNION ALL
            SELECT e.version FROM evaluations e WHERE e.version NOT IN (SELECT version FROM training_jobs WHERE version IS NOT NULL) GROUP BY e.version
        )
    """
    cursor.execute(query_count)
    total_count = cursor.fetchone()["total"]
    
    # 2. 페이징 적용된 데이터 조회
    offset = (page - 1) * page_size
    query = f"""
        SELECT 
            t.job_id,
            t.started_at as training_date,
            t.version,
            t.dataset_version as dataset,
            t.status,
            e.f1_score,
            t.progress as progress_percent
        FROM training_jobs t
        LEFT JOIN (
            SELECT version, MAX(f1_score) as f1_score 
            FROM evaluations 
            GROUP BY version
        ) e ON t.version = e.version
        WHERE t.version IS NOT NULL
        
        UNION ALL
        
        SELECT 
            NULL as job_id,
            NULL as training_date,
            e.version,
            MAX(e.dataset_version) as dataset,
            'completed' as status,
            MAX(e.f1_score) as f1_score,
            100.0 as progress_percent
        FROM evaluations e
        WHERE e.version NOT IN (SELECT version FROM training_jobs WHERE version IS NOT NULL)
        GROUP BY e.version
        
        ORDER BY training_date DESC, version DESC
        LIMIT ? OFFSET ?
    """
    cursor.execute(query, (page_size, offset))
    rows = cursor.fetchall()
    conn.close()

    history_list = []
    import hashlib
    import datetime
    for row in rows:
        t_date = row["training_date"]
        j_id = row["job_id"]
        
        display_date = t_date
        if j_id and t_date:
            h = int(hashlib.md5(j_id.encode()).hexdigest(), 16)
            try:
                dt = datetime.datetime.strptime(t_date, "%Y-%m-%dT%H:%M:%S")
                # Add minutes offset to differentiate identical started_at times for multiple identical jobs
                dt += datetime.timedelta(minutes=((h % 300) + 1))
                display_date = dt.strftime("%Y-%m-%dT%H:%M:%S")
            except:
                pass
                
        history_list.append({
            "training_date": display_date if display_date else t_date,
            "version": row["version"],
            "dataset": row["dataset"],
            "f1_score": round(row["f1_score"], 4) if row["f1_score"] is not None else None,
            "status": row["status"],
            "progress_percent": row["progress_percent"] if row["status"] != "completed" else 100.0
        })

    import math
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages
        },
        "data": history_list
    }

# =========================================
# 3. 운영 결과 기반 재학습 요청 (POST)
# =========================================
class RetrainingRequest(BaseModel):
    base_version: Optional[str] = None
    dataset_version: Optional[str] = None
    target_version: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    include_feedback: bool = False
    retraining_reason: Optional[str] = "No reason provided"
    epoch: int = 2
    batch_size: int = 8
    learning_rate: float = 0.00005

@app.post("/ai/retraining-jobs/risk-detection")
async def request_retraining_job(req: RetrainingRequest):
    now = datetime.datetime.now()
    job_id_suffix = str(uuid.uuid4())[:3].upper()
    job_id = f"retrain_{now.strftime('%Y%m%d')}_{job_id_suffix}"
    
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # base_version 자동 선택 로직 (입력 없을 시 f1_score가 가장 높은 모델)
    base_version = req.base_version
    if not base_version:
        cursor.execute("""
            SELECT version 
            FROM evaluations
            WHERE f1_score IS NOT NULL AND status = 'completed'
            ORDER BY f1_score DESC 
            LIMIT 1
        """)
        row = cursor.fetchone()
        if row and row["version"]:
            base_version = row["version"]
        else:
            # 평가 이력이 없을 경우 최신 버전으로 폴백
            cursor.execute("SELECT version FROM training_jobs WHERE status='completed' ORDER BY started_at DESC LIMIT 1")
            row_fallback = cursor.fetchone()
            if row_fallback and row_fallback["version"]:
                base_version = row_fallback["version"]
            else:
                base_version = "v1.0" # 기록이 없을 경우 v1.0 강제 지정
    
    # target_version 자동 생성 로직 (3단계 버전닝 vX.Y.Z 유지)
    target_version = req.target_version
    if not target_version:
        # 1. base_version에서 접두사(vX.Y) 추출
        parts = base_version.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
        else:
            prefix = base_version
            
        # 2. 해당 접두사로 시작하는 모든 버전 조회
        cursor.execute("SELECT version FROM training_jobs WHERE version LIKE ?", (f"{prefix}%",))
        rows = cursor.fetchall()
        
        max_patch = 0
        for row in rows:
            v = row["version"]
            v_parts = v.split('.')
            # vX.Y.Z 형태일 때 Z 값을 추출
            if len(v_parts) >= 3 and v_parts[2].isdigit():
                patch = int(v_parts[2])
                if patch > max_patch:
                    max_patch = patch
                    
        target_version = f"{prefix}.{max_patch + 1}"

    # dataset_version 자동 선택 로직
    dataset_version = req.dataset_version
    if not dataset_version:
        import glob
        import re
        files = glob.glob("./data/dataset-v*.csv")
        latest_dataset = "v1.0"
        if files:
            versions = []
            for f in files:
                m = re.search(r'dataset-(v[\d\.]+)(?:-fb-data)?\.csv', f)
                if m:
                    versions.append(m.group(1))
            if versions:
                # 숫자 기준으로 정렬 (v1.1.1 > v1.1 > v1.0)
                versions.sort(key=lambda s: [int(u) for u in s.lstrip('v').split('.')])
                latest_dataset = versions[-1]
        dataset_version = latest_dataset

    # 피드백 포함 시 접미사 추가
    if req.include_feedback:
        dataset_version = f"{dataset_version}-fb-data"

    if not os.path.exists(f"./data/dataset-{dataset_version}.csv"):
        dataset_version = "v1.0"

    # 확장된 컬럼에 재학습 정보 삽입
    cursor.execute("""
        INSERT INTO training_jobs (
            job_id, job_type, dataset_version, base_version, from_date, to_date, 
            include_feedback, retraining_reason, status,
            epoch, batch_size, learning_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, 'retrain', dataset_version, base_version, req.from_date, req.to_date,
        req.include_feedback, req.retraining_reason, "queued",
        req.epoch, req.batch_size, req.learning_rate
    ))
    conn.commit()
    conn.close()

    # 백그라운드 워커는 기존 train_api_worker.py를 그대로 재사용
    cmd = [
        "python", "src/train_api_worker.py",
        "--job_id", job_id,
        "--dataset_version", dataset_version, 
        "--epoch", str(req.epoch),
        "--batch_size", str(req.batch_size),
        "--learning_rate", str(req.learning_rate),
        "--job_type", "retrain",
        "--base_version", base_version,
        "--target_version", target_version
    ]
    
    # 메모리 파편화 방지를 위한 환경 변수 설정
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    subprocess.Popen(cmd, env=env)

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
        "progress_percent": row["progress"],
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
        
        # 토큰 사용량 저장
        usage = result.get("usage", {})
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)
        tot_tokens = usage.get("total_tokens", 0)
        
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO api_tokens (endpoint, input_tokens, output_tokens, total_tokens) VALUES (?, ?, ?, ?)", 
                       ("recommend-alternative", in_tokens, out_tokens, tot_tokens))
        conn.commit()
        conn.close()

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
        
        # 토큰 사용량 저장
        usage = result.get("usage", {})
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)
        tot_tokens = usage.get("total_tokens", 0)
        
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO api_tokens (endpoint, input_tokens, output_tokens, total_tokens) VALUES (?, ?, ?, ?)", 
                       ("classify-intent", in_tokens, out_tokens, tot_tokens))
        conn.commit()
        conn.close()
        
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

@app.get("/ai/token-usage")
async def get_token_usage():
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # 전체 합계
    cursor.execute("""
        SELECT 
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(total_tokens) as grand_total
        FROM api_tokens
    """)
    row = cursor.fetchone()
    
    # 엔드포인트별 합계
    cursor.execute("""
        SELECT 
            endpoint,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(total_tokens) as total_tokens
        FROM api_tokens
        GROUP BY endpoint
    """)
    details = []
    for r in cursor.fetchall():
        details.append({
            "endpoint": r["endpoint"],
            "input_tokens": r["input_tokens"] or 0,
            "output_tokens": r["output_tokens"] or 0,
            "total_tokens": r["total_tokens"] or 0
        })
        
    conn.close()
    
    return {
        "content_type": "json",
        "result_code": 200,
        "result_msg": "success",
        "total_usage": {
            "input_tokens": row["total_input"] or 0,
            "output_tokens": row["total_output"] or 0,
            "total_tokens": row["grand_total"] or 0
        },
        "details": details
    }

if __name__ == "__main__":
    import uvicorn
    print("\n[알림] 서비스 전용 포트로 서버를 시작합니다.")
    uvicorn.run("api:app", host="0.0.0.0", port=65001, reload=False)
