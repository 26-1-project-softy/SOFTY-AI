import sqlite3

DB_FILENAME = 'jobs.db'

def get_connection():
    conn = sqlite3.connect(DB_FILENAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. 통합 학습 상태 (Training & Retraining Jobs) 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT DEFAULT 'train',
        
        -- 공통 및 학습용 컬럼
        dataset_version TEXT,
        epoch INTEGER,
        batch_size INTEGER,
        learning_rate REAL,
        
        -- 재학습용 전용 추가 컬럼
        base_version TEXT,
        from_date TEXT,
        to_date TEXT,
        include_feedback BOOLEAN,
        retraining_reason TEXT,
        
        -- 결과 컬럼
        status TEXT,
        started_at TEXT,
        finished_at TEXT,
        model_name TEXT,
        version TEXT
    )
    """)

    # 2. 성능 평가 조회(Evaluations) 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        evaluation_id TEXT PRIMARY KEY,
        version TEXT,
        dataset_version TEXT,
        status TEXT,
        precision REAL,
        recall REAL,
        f1_score REAL,
        passed BOOLEAN
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("데이터베이스 초기화 완료(jobs.db)")
