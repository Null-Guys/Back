import os
import pandas as pd
import numpy as np
import pywt
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter


# 3) 전처리 함수: 보간 + WPT 노이즈 제거
def preprocess(df, wavelet="db4", level=3):
    df_interp = df.interpolate(method='linear').bfill().ffill()

    def wpt_denoise(col):
        data = col.values
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=level)
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric')
        for node in wp.get_level(level, 'freq'):
            new_wp[node.path] = wp[node.path].data
        rec = new_wp.reconstruct(update=False)
        if len(rec) >= len(data):
            rec = rec[:len(data)]
        else:
            rec = np.pad(rec, (0, len(data) - len(rec)), mode='edge')
        return pd.Series(rec, index=col.index)

    return df_interp.apply(wpt_denoise, axis=0)


def detect(time):
    # 1) 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILE = os.path.join(BASE_DIR, "models", "iforest_model_best.joblib")
    SCALER_FILE = os.path.join(BASE_DIR, "models", "scaler.joblib")
    TEST_CSV = os.path.join(BASE_DIR, "data", "dummy_data_1.csv")  # dummy_data_1~4.csv로 4개로 분할해놨으니 원하는 파일 선택하여 사용용

    # 2) 사용할 피처 리스트
    FEATURES = [
        "current", "voltage", "power",
        "pressure_anode_inlet", "pressure_cathode_inlet",
        "temp_anode_endplate",
        "temp_anode_inlet", "temp_cathode_dewpoint_water",
        "temp_cathode_inlet", "total_anode_stack_flow",
        "total_cathode_stack_flow"
    ]

    # 4) 모델·스케일러 로드
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # 5) 테스트 데이터 로드 & 컬럼 매핑 & 단위 변환
    df_raw = pd.read_csv(TEST_CSV)
    col_mapping = {
        "iA": "current", "U_totV": "voltage", "PW": "power",
        "P_H2_inlet": "pressure_anode_inlet", "P_Air_inlet": "pressure_cathode_inlet",
        "T_1": "temp_anode_endplate", "T_H2_inlet": "temp_anode_inlet",
        "T_3": "temp_cathode_dewpoint_water", "T_Air_inlet": "temp_cathode_inlet",
        "m_H2": "total_anode_stack_flow", "m_Air": "total_cathode_stack_flow"
    }
    df = df_raw.rename(columns=col_mapping)
    df["pressure_anode_inlet"] *= 100
    df["pressure_cathode_inlet"] *= 100
    df = df[FEATURES]

    # 6) 그라운드 트루스 라벨 생성 (10% 전압 하락 기준)
    initial_avg = df["voltage"].iloc[:200].mean()
    voltage_threshold = initial_avg * 0.90
    y_true = df["voltage"].apply(lambda v: -1 if v < voltage_threshold else 1).values

    # 7) 전처리 → 표준화
    X_proc = preprocess(df)
    X_scaled = scaler.transform(X_proc)

    # 8) anomaly score 계산
    scores = model.decision_function(X_scaled)  # 양수=정상, 음수=이상치

    # 9) 동적 threshold 계산
    contamination = model.contamination  # 예: 0.2
    thr_on_scores = np.percentile(scores, 100 * contamination)

    # 10) 최종 예측 (동적 threshold)
    # scores <= thr_on_scores 면 이상치 → -1, 그렇지 않으면 정상 → 1
    y_pred_dyn = np.where(scores <= thr_on_scores, -1, 1)

    # 11) 리스트 형태로 출력
    predictions = y_pred_dyn.tolist()

    print("predictions:", predictions)  # 결과는 리스트 형태로. 1이면 정상, -1이면 비정상

    if -1 == predictions[time]:
        return True
    else:
        return False
