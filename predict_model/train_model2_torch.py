import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# 1. 시계열 윈도우 생성
def create_windows(data, input_window=30, prediction_horizons=[10, 30], stride=1):
    print("시계열 윈도우 생성 중...")
    X, y = [], []
    feature_cols = [col for col in data.columns if '_denoised' in col and col != 'Utot (V)_denoised']
    feature_cols += ['Utot_trend', 'Utot_fluctuation']
    target_col = 'Utot (V)_denoised'
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    if not all(col in data.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in data.columns]
        raise ValueError(f"Feature columns not found: {missing_cols}")
    data_features = data[feature_cols].values
    data_target = data[target_col].values
    for i in range(0, len(data) - input_window - max(prediction_horizons) + 1, stride):
        X.append(data_features[i : i + input_window])
        y_points = [data_target[i + input_window + h - 1] for h in prediction_horizons]
        y.append(y_points)
    return np.array(X), np.array(y)

# 2. PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. 포지셔널 인코딩
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

# 4. 트랜스포머 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.2, output_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # global pooling over time dimension
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        out = self.mlp(x)
        return out

# 5. 학습/평가/시각화 함수
def plot_training_curves(train_losses, val_losses, train_maes, val_maes):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.title('MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("학습 곡선 그래프를 training_curves.png 로 저장했습니다.")

def plot_predictions(y_true, y_pred, horizons=[10, 30], sample_size=200):
    plt.figure(figsize=(15, 6))
    time_steps = np.arange(min(sample_size, len(y_true)))
    colors = ['b', 'r', 'g', 'm']
    for i, h in enumerate(horizons):
        plt.plot(time_steps, y_true[:len(time_steps), i], f'{colors[i*2]}-', label=f'실제 {h}분 후')
        plt.plot(time_steps, y_pred[:len(time_steps), i], f'{colors[i*2+1]}--', label=f'예측 {h}분 후')
    plt.title('시간에 따른 전압 예측 비교 (검증 데이터)')
    plt.xlabel('시간 단계')
    plt.ylabel('전압 (V) - Denoised')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    print("예측 비교 그래프를 prediction_comparison.png 로 저장했습니다.")

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, train_maes, val_maes = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss, running_mae = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            running_mae += torch.abs(outputs - y_batch).mean().item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_mae = running_mae / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        # Validation
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.abs(outputs - y_batch).mean().item() * X_batch.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_mae = val_mae / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    return train_losses, val_losses, train_maes, val_maes

def main():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessing_dir = os.path.abspath(os.path.join(current_dir, '..', 'preprocessing'))
    input_xlsx = os.path.join(preprocessing_dir, 'resampled_data.xlsx')
    print(f"다운샘플링된 데이터 로딩 중: {input_xlsx}")
    if not os.path.exists(input_xlsx):
        print(f"오류: {input_xlsx} 파일을 찾을 수 없습니다. 먼저 resample_data.py를 실행하세요.")
        return
    df_resampled = pd.read_excel(input_xlsx)
    # 2. 시계열 윈도우 생성 및 분할
    X, y = create_windows(df_resampled, input_window=30, prediction_horizons=[10, 30])
    if len(X) == 0:
        print("오류: 생성된 윈도우 데이터가 없습니다. 입력 데이터나 윈도우 크기를 확인하세요.")
        return
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    print(f"훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
    # PyTorch Dataset & DataLoader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 3. 모델 구축
    model = TransformerModel(
        input_dim=X_train.shape[2],
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2,
        output_dim=2
    ).to(device)
    print(model)
    # 4. 모델 학습 및 평가
    train_losses, val_losses, train_maes, val_maes = train_model(
        model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10
    )
    # 5. 시각화 (파일로 저장)
    plot_training_curves(train_losses, val_losses, train_maes, val_maes)
    # 검증 데이터 예측
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            y_pred.append(outputs.cpu().numpy())
    y_pred = np.concatenate(y_pred, axis=0)
    plot_predictions(y_val, y_pred)

if __name__ == "__main__":
    main()
