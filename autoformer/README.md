# Autoformer Fine-tuning 모델

이 프로젝트는 Hugging Face의 Autoformer 모델을 사용하여 시계열 예측을 위한 fine-tuning을 구현합니다.

## 모델 개요

- **입력**: 12개월의 과거 데이터 (8개 feature)
- **출력**: 3개월의 미래 예측값
- **Features**: temperature_max, temperature_min, temperature_range, pressure, humidity, rain, wind, sunshine_rate
- **Target**: count (이용 인원 수)

## 파일 구조

```
autoformer/
├── fine_tuning.py          # 메인 fine-tuning 코드
├── README.md              # 이 파일
├── training_curves.png    # 학습 곡선 (실행 후 생성)
├── prediction_result.png  # 예측 결과 (실행 후 생성)
└── fine_tuned_autoformer.pth  # 학습된 모델 (실행 후 생성)
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r ../requirements.txt
```

### 2. 모델 실행

```bash
cd autoformer
python fine_tuning.py
```

## 주요 기능

### 1. 데이터 전처리

- 8개 numeric feature 정규화
- 도시별 시계열 데이터 분할
- Train/Validation 데이터셋 생성

### 2. 모델 학습

- Autoformer 모델 fine-tuning
- 12개월 입력 → 3개월 예측
- AdamW optimizer 사용
- MSE Loss 함수

### 3. 결과 시각화

- 학습/검증 손실 곡선
- 예측 결과 비교 그래프

## 모델 설정

```python
# 기본 설정
input_length = 12        # 입력 시퀀스 길이 (월)
prediction_length = 3    # 예측 시퀀스 길이 (월)
num_features = 8         # feature 개수
batch_size = 16         # 배치 크기
epochs = 30             # 학습 에포크
learning_rate = 1e-4    # 학습률
```

## 사용 예시

```python
from fine_tuning import AutoformerFineTuner

# 모델 초기화
fine_tuner = AutoformerFineTuner(
    input_length=12,
    prediction_length=3,
    num_features=8
)

# 데이터 준비
feature_columns = fine_tuner.prepare_data("../data/preprocess/전처리__입력데이터_윈도우24.csv")

# 모델 학습
train_losses, val_losses = fine_tuner.train(
    batch_size=16,
    epochs=30,
    learning_rate=1e-4
)

# 예측
predictions = fine_tuner.predict(input_data)

# 모델 저장/로드
fine_tuner.save_model('fine_tuned_autoformer.pth')
fine_tuner.load_model('fine_tuned_autoformer.pth')
```

## 출력 파일

1. **training_curves.png**: 학습 과정에서의 손실 변화
2. **prediction_result.png**: 실제값과 예측값 비교
3. **fine_tuned_autoformer.pth**: 학습된 모델 가중치

## 주의사항

- GPU 사용을 권장합니다 (CUDA 지원)
- 메모리 사용량이 클 수 있으므로 배치 크기를 조정하세요
- 데이터 경로가 올바른지 확인하세요

## 문제 해결

### 메모리 부족 오류

```python
# 배치 크기 줄이기
batch_size = 8  # 또는 더 작은 값
```

### CUDA 오류

```python
# CPU 사용으로 변경
device = torch.device('cpu')
```

### 데이터 로딩 오류

- 파일 경로 확인
- CSV 파일 형식 확인
- 필요한 컬럼이 모두 있는지 확인
