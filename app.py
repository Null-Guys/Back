from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# 업로드 폴더 생성
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 수명 예측 모델 추출 칼럼명
EXTRACT_COLUMN_LIFE = ['test1', 'test2']


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

