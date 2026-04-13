# SOFTY-AI

언어 모델(LLM)이 생성하는 텍스트의 유해성(UNSAFE/SAFE)을 탐지하기 위해 모델을 파인튜닝하고, 그 성능(F1-Score 등)을 자동으로 평가하는 백그라운드 워커 기반 시스템입니다.

## 🚀 주요 기능 (Key Features)

* **모델 파인튜닝 (Training)**
  사전 학습된 모델(예: `kakaocorp/kanana-safeguard-8b`)에 유해성 분류 데이터셋을 학습시켜 특정 기준에 맞춰 파인튜닝을 진행합니다. 백그라운드 워커(`train_api_worker.py`)를 통해 OOM을 방지하며 안정적으로 학습합니다.
* **모델 성능 평가 (Evaluation)**
  파인튜닝된 모델 가중치를 로드하여 테스트 데이터셋에 대한 SAFE/UNSAFE 탐지 추론을 진행합니다. 혼동 행렬 기반의 평가지표(Precision, Recall, F1 Score)를 도출하여 기준 통과 여부를 판별합니다.
* **데이터베이스 연동**
  평가 및 학습의 진행 상태(running, completed, failed)와 채점 결과를 SQLite 기반 데이터베이스에 실시간으로 기록하고 관리합니다.
* **API 엔드포인트**
  외부에서 파인튜닝 및 평가 워커를 트리거하고 상태를 조회할 수 있는 인터페이스(REST API)를 제공합니다.

---

## 📁 디렉토리 구조 (Directory Structure)

```text
SOFTY-AI/
├── src/
│   ├── api.py               # 외부 클라이언트와의 통신을 위한 로컬 API 엔드포인트
│   ├── database.py          # SQLite DB 연결 및 상태 저장/로딩 모듈
│   ├── train_api_worker.py  # 파이토치 기반 LLM 파인튜닝 백그라운드 워커
│   └── eval_api_worker.py   # 파인튜닝 된 모델 프롬프트 평가 백그라운드 워커
├── data/                    # 모델 학습 및 평가에 사용되는 CSV 데이터 (예: dataset-v1.0.csv)
├── requirements.txt         # 파이썬 패키지 의존성 파일
└── README.md                # 프로젝트 설명 파일
```

---

## ⚙️ 요구 사항 및 설치 (Prerequisites & Installation)

본 프로젝트는 Python 3 이상의 환경과 PyTorch 생태계를 기반으로 구동됩니다. 사용 환경에 맞는 CUDA 세팅이 권장됩니다.

1. **레포지토리 클론**
   ```bash
   git clone https://github.com/26-1-project-softy/SOFTY-AI.git
   cd SOFTY-AI
   ```

2. **의존성(패키지) 설치**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏁 워커 실행 방법 (Usage)

각 모듈별로 워커를 독립적으로 터미널에서 실행할 수 있습니다. 

### 평가 워커 (Evaluation Worker)
주어진 `evaluation_id`에 대해 성능을 채점합니다.
```bash
python src/eval_api_worker.py --evaluation_id "평가ID" --version "모델버전" --dataset_version "데이터셋버전"
```
**실행 예시**:
```bash
python src/eval_api_worker.py --evaluation_id "eval-123" --version "1.0" --dataset_version "v1.0"
```

### 파인튜닝 워커 (Training Worker)
지정된 에포크(Epoch)와 배치(Batch) 설정으로 모델을 파인튜닝합니다.
```bash
python src/train_api_worker.py --job_id "작업ID" --dataset_version "데이터셋버전" --epoch 3 --batch_size 4 --learning_rate 2e-5
```

---

## 📊 데이터셋 구조 설명 (Dataset Options)

`data/` 폴더 내에 배치될 CSV 파일은 다음 컬럼 형식을 준수해야 합니다.
- `content`: 판별 및 학습에 사용할 텍스트 프롬프트
- `label`: 정답 데이터(`0` = UNSAFE 유해함, `1` = SAFE 안전함)

*초기 예시 데이터는 `dataset-v1.0.csv` 규격을 사용합니다.*