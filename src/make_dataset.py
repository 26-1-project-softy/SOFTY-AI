import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="UnSmile 데이터셋 병합 및 신규 데이터셋 생성 도구")
    parser.add_argument("--base_version", type=str, default="v1.0", help="기준이 되는 기존 데이터셋 버전 (예: v1.0)")
    parser.add_argument("--target_version", type=str, default="v1.1", help="생성할 신규 데이터셋 버전 (예: v1.1)")
    args = parser.parse_args()

    # 기존 데이터셋 경로 확인
    base_path = f"./data/dataset-{args.base_version}.csv"
    if not os.path.exists(base_path):
        print(f"[에러] 기준 데이터셋 '{base_path}'이 존재하지 않습니다.")
        return

    print("스마일게이트 UnSmile 데이터셋 다운로드 중...")
    train_url = "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_train_v1.0.tsv"
    valid_url = "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_valid_v1.0.tsv"

    try:
        train_df = pd.read_csv(train_url, sep='\t')
        valid_df = pd.read_csv(valid_url, sep='\t')
    except Exception as e:
        print(f"[에러] 외부 데이터셋 다운로드 실패: {e}")
        return

    unsmile_df = pd.concat([train_df, valid_df])

    # UnSmile 데이터셋 전처리: '문장' -> 'content', 'clean' -> 'label'
    # clean == 1 이면 SAFE (1), clean == 0 이면 UNSAFE (0)
    unsmile_df['content'] = unsmile_df['문장']
    unsmile_df['label'] = unsmile_df['clean'].astype(int)
    
    new_data = unsmile_df[['content', 'label']].dropna()
    print(f"UnSmile 데이터 파싱 완료: 총 {len(new_data)}건")
    print(new_data['label'].value_counts())

    # 기존 데이터셋 로드
    print(f"[로드] 기존 '{args.base_version}' 데이터 로드 중: {base_path}")
    base_df = pd.read_csv(base_path)
    print(f"기존 {args.base_version} 데이터 로드 완료: 총 {len(base_df)}건")

    # 데이터 병합 및 중복 제거
    merged_df = pd.concat([base_df, new_data])
    merged_df = merged_df.drop_duplicates(subset=['content']).reset_index(drop=True)
    
    print(f"병합된 {args.target_version} 데이터 생성 완료: 총 {len(merged_df)}건")
    print(merged_df['label'].value_counts())

    # 저장
    target_path = f"./data/dataset-{args.target_version}.csv"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    merged_df.to_csv(target_path, index=False)
    print(f"✅ 데이터셋 저장 완료: {target_path}")

if __name__ == "__main__":
    main()
