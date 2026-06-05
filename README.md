# 🛡️ SOFTY-AI (LLM Risk Detection & VRAM-Optimized Finetuning System)

언어 모델(LLM)이 생성하는 텍스트의 유해성(UNSAFE/SAFE)을 판별하기 위한 **파인튜닝(Finetuning)**, **실시간 추론(Inference)**, 그리고 **자동 모델 성능 평가(Evaluation)**를 지원하는 완성형 AI 백앤드 시스템입니다. 

특히, 단일 GPU(Single GPU) 환경에서 백그라운드 학습과 실시간 API 요청(추론/평가)이 경합할 때 발생하는 **VRAM 부족(Out of Memory) 문제를 완벽하게 해결하는 GPU 자원 유휴 시스템(GPU Lock & Yield)**이 내장되어 있습니다.

---

## 🏗️ 시스템 아키텍처 및 GPU 자원 관리 (GPU Coordinator)

본 시스템은 제한된 GPU 자원을 극대화하기 위해 다음과 같이 고도로 최적화된 아키텍처를 가집니다.

```mermaid
graph TD
    subgraph API Server (api.py)
        A[Client Request] --> B{Request Type}
        B -->|Inference / Eval| C[Acquire GPU Lock]
        B -->|Train / Retrain| D[Check Active Job]
    end

    subgraph GPU Lock Directory (.gpu_locks)
        C -->|Create lockfile| E[Lock Active]
    end

    subgraph Training Worker (train_api_worker.py)
        D -->|Start Training| F[Training Loop]
        F -->|Every Batch Step| G{Lock File Exists?}
        G -->|Yes| H[Yield GPU / Move to CPU]
        H -->|Free Cache| I[Update Status: paused]
        G -->|No| F
        I -->|Lock Released| J[Resume on GPU / 3x OOM Retry]
        J -->|Update Status: running| F
    end

    subgraph Evaluation / Inference (Worker Process)
        E -->|Wait 2.0s for Yield| K[Run Task on GPU]
        K -->|Process Finished| L[Release Lock]
    end
```

### 1. 동적 GPU 자원 양보 및 복구 시스템 (Pause & Resume)
* **API 자원 독점 방지**: 실시간 추론 및 평가 요청이 유입되면 `.gpu_locks` 디렉토리에 락 파일이 생성됩니다.
* **학습 스레드 CPU 대피**: 백그라운드 학습 프로세스(`train_api_worker.py`)는 매 배치 스텝마다 이 디렉토리를 감시합니다. 락이 감지되는 즉시 **가중치와 옵티마이저 가중치를 CPU 메모리로 마이그레이션(`model.cpu()`)**하고 `torch.cuda.empty_cache()`를 실행하여 GPU VRAM을 완전히 비워줍니다.
* **OOM 방어 재시도 복구**: 실시간 요청 처리가 끝나 락이 풀리면, 학습 프로세스는 모델을 다시 GPU로 로드합니다. 이때 일시적인 VRAM 파편화로 인한 OOM을 방지하기 위해 **최대 3회의 복구 재시도(Retry) 및 CUDA 캐시 플러시 매커니즘**이 내장되어 있습니다.

### 2. OS 레벨 VRAM 자원 완벽 회수 (Inference Isolation)
* 파이썬 내에서 모델 가중치를 로드하고 언로드하더라도 PyTorch 캐시나 CUDA 컨텍스트로 인해 GPU 메모리가 완전히 반환되지 않을 수 있습니다.
* 이를 해결하기 위해 실시간 추론 요청이 들어올 때마다 **OS 서브프로세스(`inference_worker.py`)를 독립 실행**하여 추론을 마친 후 프로세스 종료와 함께 **OS 레벨에서 GPU 메모리를 100% 강제 해제**합니다.

---

## 📁 디렉토리 구조 (Directory Structure)

```text
SOFTY-AI/
├── src/
│   ├── api.py               # 외부 인터페이스를 제공하는 FastAPI 메인 웹 서버
│   ├── database.py          # SQLite DB 연결 및 상태 저장/로딩 모듈
│   ├── train_api_worker.py  # GPU 자원 양보 로직이 포함된 PyTorch 학습 워커
│   ├── eval_api_worker.py   # 모델 성능 채점 및 평가 자동화 워커 (GPU Lock 요청)
│   └── inference_worker.py  # 단발성 프로세스로 실행되어 VRAM 유출을 차단하는 추론 워커
├── data/                    # 모델 학습 및 평가에 사용되는 CSV 데이터 보관 폴더
├── requirements.txt         # 파이썬 의존성 패키지 리스트
├── .gpu_locks/              # 실시간 GPU 자원 조율용 락 파일이 생성되는 임시 디렉토리
└── README.md                # 본 프로젝트 설명서
```

---

## ⚙️ 요구 사항 및 설치 (Prerequisites & Installation)

본 시스템은 Python 3.9+ 이상의 환경과 CUDA 가용한 PyTorch 생태계를 기반으로 작동합니다.

1. **레포지토리 클론**
   ```bash
   git clone https://github.com/26-1-project-softy/SOFTY-AI.git
   cd SOFTY-AI
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 변수 파일 (`.env`) 설정**
   루트 경로에 `.env` 파일을 생성하고 외부 LLM 호출용 정보 등을 입력합니다.
   ```env
   EXTERNAL_API_URL=http://your-llm-api-url:port/v1/chat/completions
   TEAM_KEY=your_secret_team_key
   ```

---

## 📊 데이터베이스 스키마 및 규격 (Database)

시스템 실행 시 자동으로 SQLite 데이터베이스(`jobs.db`)가 생성되며 아래 3가지 테이블을 관리합니다.

1. **`training_jobs`** (학습 상태 및 히스토리 관리)
   * `job_id` (PK), `job_type` (`train` / `retrain`), `status` (`queued`, `running`, `paused`, `completed`, `failed`), `epoch`, `batch_size`, `learning_rate`, `progress`, `version` 등
2. **`evaluations`** (평가 결과 및 지표 기록)
   * `evaluation_id` (PK), `version` (모델 버전), `dataset_version`, `status`, `precision`, `recall`, `f1_score`, `passed` 등
3. **`api_tokens`** (API 사용량 추적)
   * `endpoint`, `input_tokens`, `output_tokens`, `total_tokens`, `created_at` 등

---

## 🏁 워커 및 API 서버 실행 방법 (Usage)

### 1. API 서버 구동 (FastAPI)
포트 `65001`번을 기본값으로 서비스를 구동합니다.
```bash
python src/api.py
```

### 2. 백그라운드 워커 개별 수동 실행 (디버깅용)
* **학습 워커 (Training)**:
  ```bash
  python src/train_api_worker.py --job_id "train_example" --dataset_version "v1.0" --epoch 4 --batch_size 8 --learning_rate 0.00005
  ```
* **평가 워커 (Evaluation)**:
  ```bash
  python src/eval_api_worker.py --evaluation_id "eval_example" --version "v1.2" --dataset_version "v1.0"
  ```
* **일시적 추론 워커 (Inference)**:
  ```bash
  python src/inference_worker.py "테스트할 문장 내용" "v1.2"
  ```

---

## 🔌 API 엔드포인트 명세 (API Endpoints)

| 엔드포인트 | 메서드 | 설명 |
| :--- | :---: | :--- |
| `/health` | `GET` | API 서버의 헬스체크 및 실행 상태 반환 |
| `/ai/training-jobs/risk-detection` | `POST` | 사전 정의된 모델에 유해 데이터 학습 요청 (중복 방지 체크) |
| `/ai/training-jobs/status` | `GET` | 진행 중이거나 마지막 완료된 학습 진행도(`progress`) 및 상태 조회 |
| `/ai/training-jobs/history` | `GET` | 과거 학습 히스토리 전체 조회 (페이지네이션 지원) |
| `/ai/training-jobs/retrain` | `POST` | 추가된 사용자 피드백 데이터를 활용한 모델 재학습(Retraining) 트리거 |
| `/ai/evaluate/risk-detection` | `POST` | 특정 가중치 버전에 대한 유해성 탐지 채점(Evaluation) 평가 시작 |
| `/ai/evaluate/result` | `GET` | 평가 ID 기반 혼동 행렬 지표(F1-score, Precision 등) 상세 결과 조회 |
| `/ai/inference/risk-detection` | `POST` | 텍스트 유해성 탐지(SAFE/UNSAFE) 실시간 분류 추론 (GPU Lock 및 최적 버전 동적 적용) |

---

## 📈 데이터셋 및 학습 데이터 포맷 (Dataset Options)

`data/` 폴더 내에 저장될 모델 파인튜닝/평가용 CSV 데이터셋은 다음과 같은 형식이어야 합니다.

```csv
content,label
"사용자가 보낸 질문 예시 텍스트입니다.",1
"유해하거나 심각한 분쟁 요소가 섞인 비공개 데이터셋...",0
```
* **label**: `0` = UNSAFE (유해함 / 위험), `1` = SAFE (안전함)
* 파일명 형식 권장: `dataset-{version}.csv` (예: `dataset-v1.0.csv`)