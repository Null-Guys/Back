from flask import Flask, request, jsonify
import pandas as pd
import os
from predict_model.predict_from_time import predict
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 업로드 폴더 생성
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 수명 예측 모델 추출 칼럼명
EXTRACT_COLUMN_LIFE = ['DoutAIR', 'I', 'J', 'TinH2', 'DWAT', 'DoutH2', 'PoutAIR', 'HrAIRFC']

# 초기 전압 값
INIT_U = 3.4


@app.route('/')
def index():
    return 'Hello, World!'


@app.route('/upload', methods=['POST'])
def upload():
    # 파일 수신 확인
    if 'file' not in request.files:
        return jsonify({'error': 'Did not receive the file'}), 400

    # 파일 이름 확인
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File name is empty'}), 400

    try:
        # 확장자를 제외한 파일 이름 추출
        file_name = os.path.splitext(file.filename)[0]
        save_path = os.path.join(UPLOAD_FOLDER, f'{file_name}_filtered.csv')

        # pandas로 CSV 읽기
        data_file = pd.read_csv(file)

        # 필요한 컬럼 추출
        filtered_data_file = data_file[EXTRACT_COLUMN_LIFE]

        # 저장
        filtered_data_file.to_csv(save_path, index=False)

        return jsonify({'message': 'Uploading and saving is success', 'saved_path': save_path}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/information', methods=['GET'])
def information():
    time = request.args.get('time', type=int)

    # 파라미터 수신 확인
    if time is None:
        return jsonify({'error': 'filename and time are required'}), 400

    # 예측 출력 값, 이 부분 모델 연동으로 바꿔야 함
    d = predict(time)
    Utot = d[0]

    response = {
        'SOH': calculate_soh(Utot[1], INIT_U),
        'VDR': calculate_vdr(INIT_U, Utot[1], 30),
        'RUL': calculate_rul(Utot, 3.3),
        'STT': calculate_stt(calculate_soh(Utot[1], INIT_U)),
        'isBreak': False,
        'time': time + 30,
        'data': d[1]
    }

    # 10개 데이터 리스트에 추가
    #    for _, row in sliced_df.iterrows():
    #        item = {
    #            'DoutAIR': row.get('DoutAIR'),
    #            'I': row.get('I'),
    #            'J': row.get('J'),
    #            'TinH2': row.get('TinH2'),
    #            'DWAT': row.get('DWAT'),
    #            'DoutH2': row.get('DoutH2'),
    #            'PoutAIR': row.get('PoutAIR'),
    #            'HrAIRFC': row.get('HrAIRFC')
    #        }
    #        response['data'].append(item)

    return jsonify(response)


def calculate_soh(u_predicted: float, u_initial: float) -> float:
    if u_initial == 0:
        raise ValueError("'u_initial' cannot be zero")
    soh = (u_predicted / u_initial) * 100
    return round(soh, 2)


def calculate_vdr(u_t: float, u_t_n: float, delta_t: float) -> float:
    if delta_t == 0:
        raise ValueError("'delta_t' cannot be zero")
    degradation_rate = (u_t - u_t_n) / delta_t
    return round(degradation_rate, 6)


def calculate_rul(u_predicted_list: list, u_limit: float) -> int:
    if u_predicted_list[0] <= u_limit:
        return 10
    if u_predicted_list[1] <= u_limit:
        return 30
    return -1


def calculate_stt(soh: float) -> str:
    if soh > 90:
        return "정상 (유지 운전)"
    elif 80 <= soh <= 90:
        return "경고 (정기 점검 권장)"
    else:
        return "열화 (교체 또는 분해 점검 필요)"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
