import pandas as pd
import numpy as np
import torch
import os
import sys

from predict_model.train_model2_torch import PositionalEncoding, TransformerModel


def create_window_from_minute(data, start_minute, input_window=30):
    """지정된 시작 시간(분)부터 30분 데이터 윈도우 생성"""
    print(f"시작 시간 {start_minute}분부터 {input_window}분 데이터 윈도우 생성 중...")

    # 시작 시간을 시간 단위로 변환
    start_hour = start_minute / 60

    # 데이터에서 가장 가까운 시간 포인트 찾기
    time_diffs = abs(data['Time (h)'] - start_hour)
    start_idx = time_diffs.argmin()

    if start_idx + input_window > len(data):
        raise ValueError(f"시작 시간부터 {input_window}분 데이터를 추출할 수 없습니다. 데이터가 충분하지 않습니다.")

    # 입력 특성 선택
    feature_cols = [col for col in data.columns if '_denoised' in col and col != 'Utot (V)_denoised']
    feature_cols += ['Utot_trend', 'Utot_fluctuation']

    if not all(col in data.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in data.columns]
        raise ValueError(f"필요한 특성 컬럼을 찾을 수 없습니다: {missing_cols}")

    # 데이터 추출
    window_data = data.iloc[start_idx:start_idx + input_window][feature_cols].values

    if len(window_data) < input_window:
        raise ValueError(f"시작 시간부터 {input_window}분 데이터를 추출할 수 없습니다. 데이터가 충분하지 않습니다.")

    ####
    print("사용된 feature columns:", feature_cols)  # ✅ 1번 목적

    # 2. 데이터 추출
    window_df = data.iloc[start_idx:start_idx + input_window][feature_cols]

    # 3. 컬럼명과 값 매핑 결과 생성
    feature_data = window_df.to_dict(orient='records')

    print("사용된 feature 데이터:", feature_data)
    ###

    return [torch.FloatTensor(window_data).unsqueeze(0), feature_data]  # 배치 차원 추가


def predict(time):
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best_model.pt')
    data_path = os.path.join(current_dir, 'resampled_data.xlsx')

    # 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    if not os.path.exists(data_path):
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    # 시작 시간 입력 받기
    print("시작 시간을 분 단위로 입력하세요 (예: 27): ")
    try:
        start_minute = time
        if start_minute < 0:
            raise ValueError("시작 시간은 0 이상이어야 합니다.")
    except ValueError as e:
        print(f"올바른 숫자를 입력해주세요: {e}")
        return

    try:
        # 데이터 로드
        df = pd.read_excel(data_path)

        # 모델 로드
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = len([col for col in df.columns if
                         '_denoised' in col and col != 'Utot (V)_denoised']) + 2  # denoised 특성 + trend + fluctuation
        model = TransformerModel(input_dim=input_dim)  # TransformerModel 인스턴스 생성
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 입력 데이터 생성
        d = create_window_from_minute(df, start_minute)
        X = d[0]
        X = X.to(device)

        # 예측 수행
        print("예측 수행 중...")
        with torch.no_grad():
            predictions = model(X)
        predictions = predictions.cpu().numpy()[0]  # 배치 차원 제거

        # 결과 출력
        print("\n예측 결과:")
        print(f"10분 후 예측 전압: {predictions[0]:.4f} V")
        print(f"30분 후 예측 전압: {predictions[1]:.4f} V")

    except ValueError as e:
        print(f"오류 발생: {e}")
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
    return [[float(predictions[0]), float(predictions[1])], d[1]]


if __name__ == "__main__":
    main()
