import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoformerConfig, AutoformerModel, AutoformerForPrediction
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
import datetime
from dateutil.relativedelta import relativedelta

# matplotlib 한글 폰트 설정
import matplotlib.font_manager as fm

# macOS에서 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"  # macOS 기본 한글 폰트
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 폰트가 없을 경우 대체 폰트 설정
try:
    # AppleGothic 폰트 확인
    fm.findfont("AppleGothic")
except:
    # 대체 폰트들 시도
    alternative_fonts = ["NanumGothic", "Malgun Gothic", "Gulim", "Dotum"]
    for font in alternative_fonts:
        try:
            plt.rcParams["font.family"] = font
            fm.findfont(font)
            print(f"Using font: {font}")
            break
        except:
            continue
    else:
        # 모든 한글 폰트가 없으면 기본 폰트 사용
        print("Warning: No Korean font found. Using default font.")
        plt.rcParams["font.family"] = "DejaVu Sans"

warnings.filterwarnings("ignore")


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        input_length=12,
        prediction_length=3,
        feature_columns=None,
        target_column="count",
    ):
        self.data = data
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.feature_columns = feature_columns
        self.target_column = target_column

        self.samples = []
        for city in data["city"].unique():
            city_data = data[data["city"] == city].sort_values("year_month")
            if len(city_data) < input_length + prediction_length:
                continue
            features = city_data[feature_columns].values
            targets = city_data[target_column].values
            for i in range(len(city_data) - input_length - prediction_length + 1):
                past_values = targets[i : i + input_length]
                future_values = targets[
                    i + input_length : i + input_length + prediction_length
                ]
                past_time_features = features[i : i + input_length]
                future_time_features = features[
                    i + input_length : i + input_length + prediction_length
                ]
                past_observed_mask = np.ones_like(past_values, dtype=np.float32)
                self.samples.append(
                    {
                        "past_values": torch.tensor(past_values, dtype=torch.float32),
                        "future_values": torch.tensor(
                            future_values, dtype=torch.float32
                        ),
                        "past_time_features": torch.tensor(
                            past_time_features, dtype=torch.float32
                        ),
                        "future_time_features": torch.tensor(
                            future_time_features, dtype=torch.float32
                        ),
                        "past_observed_mask": torch.tensor(
                            past_observed_mask, dtype=torch.float32
                        ),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AutoformerFineTuner:
    def __init__(
        self,
        model_name="huggingface/autoformer-tourism-monthly",
        input_length=12,
        prediction_length=3,
        num_features=8,
    ):
        self.model_name = model_name
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 설정 - 처음부터 생성
        self.config = AutoformerConfig()

        # 기본 설정
        self.config.context_length = input_length
        self.config.prediction_length = prediction_length
        self.config.num_dynamic_real_features = num_features
        self.config.num_static_categorical_features = 0
        self.config.num_static_real_features = 0
        self.config.num_time_features = 0
        self.config.num_dynamic_categorical_features = 0
        self.config.input_size = 1  # target variable (count)
        self.config.feature_size = 11  # 실제 입력 feature 차원

        # Autoformer 특정 설정
        self.config.d_model = 64
        self.config.encoder_attention_heads = 8
        self.config.decoder_attention_heads = 8
        self.config.encoder_layers = 2
        self.config.decoder_layers = 2
        self.config.encoder_ffn_dim = 256
        self.config.decoder_ffn_dim = 256
        self.config.dropout = 0.1
        self.config.activation_dropout = 0.1
        self.config.attention_dropout = 0.1
        self.config.lags_sequence = [0]

        # 모델 초기화
        self.model = AutoformerForPrediction(self.config)
        self.model.to(self.device)

        # 스케일러 초기화
        self.scaler = StandardScaler()

    def prepare_data(self, data_path):
        """데이터 준비 및 전처리"""
        print("데이터 로딩 중...")
        df = pd.read_csv(data_path)

        # feature 컬럼 정의
        feature_columns = [
            "temperature_max",
            "temperature_min",
            "temperature_range",
            "pressure",
            "humidity",
            "rain",
            "wind",
            "sunshine_rate",
        ]

        # 데이터 정규화
        print("데이터 정규화 중...")
        df_normalized = df.copy()
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])

        # train/validation 분할
        cities = df["city"].unique()
        train_cities, val_cities = train_test_split(
            cities, test_size=0.2, random_state=42
        )

        # city 분할 결과 출력
        print(f"\n=== City Split Results ===")
        print(f"Total cities: {len(cities)}")
        print(f"Train cities ({len(train_cities)}): {sorted(train_cities)}")
        print(f"Validation cities ({len(val_cities)}): {sorted(val_cities)}")
        print("=" * 50)

        # city 정보 저장
        self.train_cities = train_cities
        self.val_cities = val_cities
        self.all_cities = cities

        train_data = df_normalized[df_normalized["city"].isin(train_cities)]
        val_data = df_normalized[df_normalized["city"].isin(val_cities)]

        # 데이터셋 생성
        print("데이터셋 생성 중...")
        self.train_dataset = TimeSeriesDataset(
            train_data,
            input_length=self.input_length,
            prediction_length=self.prediction_length,
            feature_columns=feature_columns,
        )

        self.val_dataset = TimeSeriesDataset(
            val_data,
            input_length=self.input_length,
            prediction_length=self.prediction_length,
            feature_columns=feature_columns,
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        return feature_columns

    def train(self, batch_size=32, epochs=100, learning_rate=3e-5):
        """모델 학습"""
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        print(f"학습 시작 (Device: {self.device})")

        for epoch in range(epochs):
            # 학습
            self.model.train()
            train_loss = 0
            train_batches = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"
            ):
                for k in batch:
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(
                    past_values=batch["past_values"],
                    past_time_features=batch["past_time_features"],
                    future_time_features=batch["future_time_features"],
                    past_observed_mask=batch["past_observed_mask"],
                    future_values=batch["future_values"],
                )
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1

            # 검증
            self.model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"
                ):
                    for k in batch:
                        batch[k] = batch[k].to(self.device)
                    outputs = self.model(
                        past_values=batch["past_values"],
                        past_time_features=batch["past_time_features"],
                        future_time_features=batch["future_time_features"],
                        past_observed_mask=batch["past_observed_mask"],
                        future_values=batch["future_values"],
                    )
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_batches += 1
            avg_train_loss = train_loss / train_batches
            avg_val_loss = val_loss / val_batches
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {avg_val_loss:.6f}")
            print("-" * 50)
        self.plot_training_curves(train_losses, val_losses)
        return train_losses, val_losses

    def predict(self, past_values, past_time_features, future_time_features):
        """예측 수행"""
        self.model.eval()

        with torch.no_grad():
            # 입력 데이터를 Autoformer 형식에 맞게 변환
            past_values = torch.FloatTensor(past_values).unsqueeze(0).to(self.device)
            past_time_features = (
                torch.FloatTensor(past_time_features).unsqueeze(0).to(self.device)
            )
            future_time_features = (
                torch.FloatTensor(future_time_features).unsqueeze(0).to(self.device)
            )
            past_observed_mask = torch.ones_like(past_values).to(self.device)

            # Autoformer의 generate 메서드 사용 (기본 설정)
            outputs = self.model.generate(
                past_values=past_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                past_observed_mask=past_observed_mask,
            )

            # 예측 결과 추출 - 첫 번째 시퀀스만 사용
            predictions = outputs.sequences.cpu().numpy()[0]

            # 예측 결과가 예상 길이와 다른 경우 처리
            if len(predictions.shape) > 1:
                # 여러 시퀀스가 있는 경우 첫 번째 시퀀스만 사용
                predictions = predictions[0]

            # 예측 길이가 prediction_length와 다른 경우 조정
            if len(predictions) != self.prediction_length:
                print(
                    f"Warning: Expected {self.prediction_length} predictions, got {len(predictions)}"
                )
                if len(predictions) > self.prediction_length:
                    predictions = predictions[: self.prediction_length]
                else:
                    # 부족한 경우 마지막 값으로 패딩
                    last_val = predictions[-1] if len(predictions) > 0 else 0
                    predictions = np.append(
                        predictions,
                        [last_val] * (self.prediction_length - len(predictions)),
                    )

        return predictions

    def plot_training_curves(self, train_losses, val_losses):
        """학습 곡선 시각화"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", color="blue")
        plt.plot(val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
        plt.show()

    def save_model(self, path):
        """모델 저장"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "scaler": self.scaler,
            },
            path,
        )
        print(f"모델이 {path}에 저장되었습니다.")

    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]
        print(f"모델이 {path}에서 로드되었습니다.")

    def predict_for_city_month(self, city, year_month, data_path):
        """특정 도시와 연월에 대한 예측 수행"""
        print(f"\n=== Prediction for {city} at {year_month} ===")

        # 데이터 로드
        df = pd.read_csv(data_path)

        # 해당 도시의 데이터 필터링
        city_data = df[df["city"] == city].sort_values("year_month")

        if len(city_data) == 0:
            print(f"Error: City '{city}' not found in data")
            return None

        # year_month 인덱스 찾기
        target_idx = city_data[city_data["year_month"] == year_month].index

        # 데이터에 해당 year_month가 있는 경우 (검증용)
        if len(target_idx) > 0:
            target_idx = target_idx[0]
            city_data_idx = city_data.index.get_loc(target_idx)

            # 충분한 과거 데이터가 있는지 확인
            if city_data_idx < self.input_length:
                print(f"Error: Not enough past data for {city} at {year_month}")
                print(f"Need at least {self.input_length} months of past data")
                return None

            # feature 컬럼 정의
            feature_columns = [
                "temperature_max",
                "temperature_min",
                "temperature_range",
                "pressure",
                "humidity",
                "rain",
                "wind",
                "sunshine_rate",
            ]

            # 데이터 정규화 (train 데이터의 스케일러 사용)
            city_data_normalized = city_data.copy()
            city_data_normalized[feature_columns] = self.scaler.transform(
                city_data[feature_columns]
            )

            # 과거 데이터 추출 (12개월)
            start_idx = city_data_idx - self.input_length
            past_data = city_data_normalized.iloc[start_idx:city_data_idx]

            # 미래 데이터 추출 (3개월) - 실제 값과 비교용
            future_data = city_data.iloc[
                city_data_idx : city_data_idx + self.prediction_length
            ]

            # 입력 데이터 준비
            past_values = past_data["count"].values
            past_time_features = past_data[feature_columns].values
            future_time_features = future_data[feature_columns].values
            true_future_values = future_data["count"].values

            # 예측 수행
            predictions = self.predict(
                past_values, past_time_features, future_time_features
            )

            # 결과 출력
            print(f"Past 12 months (count): {past_values}")
            print(f"Actual future 3 months: {true_future_values}")
            print(f"Predicted future 3 months: {predictions}")

            # 예측 정확도 계산
            mse = np.mean((true_future_values - predictions) ** 2)
            mae = np.mean(np.abs(true_future_values - predictions))
            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Mean Absolute Error: {mae:.2f}")

            # 시각화
            self.plot_prediction_result(
                past_values,
                true_future_values,
                predictions,
                city,
                year_month,
                has_actual=True,
            )

            return {
                "city": city,
                "year_month": year_month,
                "past_values": past_values,
                "actual_future": true_future_values,
                "predicted_future": predictions,
                "mse": mse,
                "mae": mae,
                "prediction_type": "validation",
            }

        # 데이터에 해당 year_month가 없는 경우 (미래 예측)
        else:
            print(
                f"Note: {year_month} not found in data. Using latest data for future prediction."
            )

            # 가장 최근 데이터의 인덱스
            latest_idx = len(city_data) - 1

            # 충분한 과거 데이터가 있는지 확인
            if latest_idx < self.input_length - 1:
                print(f"Error: Not enough past data for {city}")
                print(f"Need at least {self.input_length} months of past data")
                return None

            # feature 컬럼 정의
            feature_columns = [
                "temperature_max",
                "temperature_min",
                "temperature_range",
                "pressure",
                "humidity",
                "rain",
                "wind",
                "sunshine_rate",
            ]

            # 데이터 정규화 (train 데이터의 스케일러 사용)
            city_data_normalized = city_data.copy()
            city_data_normalized[feature_columns] = self.scaler.transform(
                city_data[feature_columns]
            )

            # 과거 데이터 추출 (12개월) - 가장 최근 데이터부터
            start_idx = latest_idx - self.input_length + 1
            past_data = city_data_normalized.iloc[start_idx : latest_idx + 1]

            # 입력 데이터 준비
            past_values = past_data["count"].values
            past_time_features = past_data[feature_columns].values

            # 미래 feature는 과거의 평균값을 사용하거나, 마지막 값들을 반복 사용
            # 여기서는 마지막 3개월의 feature 평균을 사용
            last_features = past_time_features[-3:]  # 마지막 3개월
            future_time_features = np.tile(
                np.mean(last_features, axis=0), (self.prediction_length, 1)
            )

            # 예측 수행
            predictions = self.predict(
                past_values, past_time_features, future_time_features
            )

            # 결과 출력
            print(f"Latest 12 months (count): {past_values}")
            print(f"Predicted future 3 months: {predictions}")
            print(
                f"Note: This is a true future prediction (no actual values for comparison)"
            )

            # 시각화
            self.plot_prediction_result(
                past_values, None, predictions, city, year_month, has_actual=False
            )

            return {
                "city": city,
                "year_month": year_month,
                "past_values": past_values,
                "actual_future": None,
                "predicted_future": predictions,
                "mse": None,
                "mae": None,
                "prediction_type": "future",
            }

    def plot_prediction_result(
        self, past_values, true_future, predictions, city, year_month, has_actual=True
    ):
        """예측 결과 시각화"""
        plt.figure(figsize=(15, 6))

        # 과거 데이터의 year_month 생성 (12개월)
        # year_month에서 12개월 전부터 시작
        if "-" in year_month:
            year, month = map(int, year_month.split("-"))
        else:
            year, month = map(int, year_month.split("/"))

        target_date = datetime.datetime(year, month, 1)

        # 과거 12개월의 날짜 생성
        past_dates = []
        for i in range(11, -1, -1):  # 11개월 전부터 현재까지
            past_date = target_date - relativedelta(months=i)
            past_dates.append(past_date.strftime("%Y-%m"))

        # 미래 3개월의 날짜 생성
        future_dates = []
        for i in range(1, 4):  # 다음 3개월
            future_date = target_date + relativedelta(months=i)
            future_dates.append(future_date.strftime("%Y-%m"))

        # 과거 데이터 (12개월) - 선으로 연결
        plt.plot(
            past_dates,
            past_values,
            "o-",
            label="Past Values",
            color="blue",
            linewidth=2,
            markersize=6,
        )

        # 과거 데이터 포인트에 값 표시
        for i, (date, value) in enumerate(zip(past_dates, past_values)):
            plt.annotate(
                f"{value:.0f}",
                (date, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="blue",
            )

        if has_actual and true_future is not None:
            # 실제 미래 데이터 (3개월) - 과거와 연결
            actual_values = [past_values[-1]] + list(true_future)
            actual_dates = [past_dates[-1]] + future_dates
            plt.plot(
                actual_dates,
                actual_values,
                "s-",
                label="Actual Future",
                color="green",
                linewidth=2,
                markersize=8,
            )

            # 실제 미래 데이터 포인트에 값 표시 (과거 마지막 값 제외)
            for i, (date, value) in enumerate(zip(future_dates, true_future)):
                plt.annotate(
                    f"{value:.0f}",
                    (date, value),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color="green",
                )

        # 예측 미래 데이터 (3개월) - 과거와 연결
        pred_values = [past_values[-1]] + list(predictions)
        pred_dates = [past_dates[-1]] + future_dates
        plt.plot(
            pred_dates,
            pred_values,
            "^--",
            label="Predicted Future",
            color="red",
            linewidth=2,
            markersize=8,
        )

        # 예측 미래 데이터 포인트에 값 표시 (과거 마지막 값 제외)
        for i, (date, value) in enumerate(zip(future_dates, predictions)):
            plt.annotate(
                f"{value:.0f}",
                (date, value),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
                color="red",
            )

        plt.xlabel("Year-Month")
        plt.ylabel("Number of Users")
        plt.title(f"Prediction Results for {city} at {year_month}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # x축 레이블 회전
        plt.xticks(rotation=45)

        # 구분선 추가 (예측 시작점)
        plt.axvline(x=past_dates[-1], color="gray", linestyle="--", alpha=0.7)
        plt.text(
            past_dates[-1],
            plt.ylim()[1] * 0.9,
            "Prediction Start",
            rotation=90,
            alpha=0.7,
        )

        plt.tight_layout()
        plt.savefig(f"prediction_{city}_{year_month}.png", dpi=300, bbox_inches="tight")
        plt.show()

    def interactive_prediction(self, data_path):
        """사용자 인터랙티브 예측"""
        print("\n=== Interactive Prediction ===")
        print("Available cities:", sorted(self.all_cities))

        while True:
            try:
                print("\n--- Prediction Options ---")
                print("1. Predict for existing data (validation)")
                print("2. Predict for future (using latest data)")
                print("3. Back to main menu")

                pred_choice = input("\nEnter your choice (1-3): ").strip()

                if pred_choice == "1":
                    # 기존 데이터에 대한 예측 (검증용)
                    city = input("Enter city name: ").strip()
                    if city not in self.all_cities:
                        print(
                            f"City '{city}' not found. Available cities: {sorted(self.all_cities)}"
                        )
                        continue

                    year_month = input(
                        "Enter year-month (YYYY-MM format, e.g., 2024-01): "
                    ).strip()

                    # 예측 수행
                    result = self.predict_for_city_month(city, year_month, data_path)

                    if result:
                        print(f"\nPrediction completed for {city} at {year_month}")
                        if result["prediction_type"] == "validation":
                            print(f"MSE: {result['mse']:.2f}, MAE: {result['mae']:.2f}")

                elif pred_choice == "2":
                    # 미래 예측 (최신 데이터 사용)
                    city = input("Enter city name: ").strip()
                    if city not in self.all_cities:
                        print(
                            f"City '{city}' not found. Available cities: {sorted(self.all_cities)}"
                        )
                        continue

                    future_month = input(
                        "Enter future year-month (YYYY-MM format, e.g., 2025-01): "
                    ).strip()

                    # 예측 수행
                    result = self.predict_for_city_month(city, future_month, data_path)

                    if result:
                        print(
                            f"\nFuture prediction completed for {city} at {future_month}"
                        )
                        print(
                            "Note: This is a true future prediction using the latest available data."
                        )

                elif pred_choice == "3":
                    break

                else:
                    print("잘못된 선택입니다. 1-3 중에서 선택해주세요.")

                # 계속할지 묻기
                continue_pred = (
                    input("\nContinue with another prediction? (y/n): ").strip().lower()
                )
                if continue_pred != "y":
                    break

            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


def main():
    """메인 함수"""
    print("Autoformer Fine-tuning 시작")

    # 데이터 경로
    data_path = "../data/preprocess/전처리__입력데이터_윈도우24.csv"

    # Fine-tuner 초기화
    fine_tuner = AutoformerFineTuner()

    # 데이터 준비
    feature_columns = fine_tuner.prepare_data(data_path)

    # 모델이 이미 존재하는지 확인
    model_path = "fine_tuned_autoformer.pth"
    if os.path.exists(model_path):
        print(f"\n기존 모델을 발견했습니다: {model_path}")
        load_choice = input("기존 모델을 로드하시겠습니까? (y/n): ").strip().lower()
        if load_choice == "y":
            fine_tuner.load_model(model_path)
            print("기존 모델을 로드했습니다.")
        else:
            print("새로운 모델을 학습합니다.")
            # 학습 수행
            train_losses, val_losses = fine_tuner.train(epochs=100)
            # 모델 저장
            fine_tuner.save_model(model_path)
    else:
        print("새로운 모델을 학습합니다.")
        # 학습 수행
        train_losses, val_losses = fine_tuner.train(epochs=100)
        # 모델 저장
        fine_tuner.save_model(model_path)

    print("Fine-tuning 완료!")
    print(f"모델이 {model_path}에 저장되었습니다.")

    # 인터랙티브 예측 모드 시작
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 60)

    while True:
        try:
            print("\n=== Available Options ===")
            print("1. Predict for specific city and year-month")
            print("2. Show city split information")
            print("3. Retrain model")
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == "1":
                # 예측 수행
                fine_tuner.interactive_prediction(data_path)

            elif choice == "2":
                # city 분할 정보 출력
                print(f"\n=== City Split Information ===")
                print(f"Total cities: {len(fine_tuner.all_cities)}")
                print(
                    f"Train cities ({len(fine_tuner.train_cities)}): {sorted(fine_tuner.train_cities)}"
                )
                print(
                    f"Validation cities ({len(fine_tuner.val_cities)}): {sorted(fine_tuner.val_cities)}"
                )

            elif choice == "3":
                # 모델 재학습
                print("\n=== Retraining Model ===")
                retrain_choice = (
                    input("정말로 모델을 재학습하시겠습니까? (y/n): ").strip().lower()
                )
                if retrain_choice == "y":
                    train_losses, val_losses = fine_tuner.train(epochs=30)
                    fine_tuner.save_model(model_path)
                    print("모델 재학습이 완료되었습니다.")

            elif choice == "4":
                print("프로그램을 종료합니다.")
                break

            else:
                print("잘못된 선택입니다. 1-4 중에서 선택해주세요.")

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue


if __name__ == "__main__":
    main()
