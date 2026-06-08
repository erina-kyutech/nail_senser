# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 設定
# =========================================================
RANDOM_SEED = 42

# 画像サイズ（concat後）
IMG_H = 150
IMG_W = 290
IMG_C = 3

# 学習条件
BATCH_SIZE = 32
EPOCHS = 10
VAL_RATIO = 0.15

# 力の正規化
NORMAL_FORCE_NORMALIZE = 10.0
SHEAR_FORCE_NORMALIZE = 5.0

# 被験者名
SUBJECT_NAME = "ifuku"

# テスト分割
TRAINVAL_START = 1
TRAINVAL_END = 166
TEST_START = 167
TEST_END = 180

# 入力モード
IMG_MODE = "rgb"

# ===== 黒塗り設定 =====
MASK_LEFT_PIXELS = 40
MASK_RIGHT_PIXELS = 0
MASK_TOP_PIXELS = 0
MASK_BOTTOM_PIXELS = 0

# 保存先（別フォルダ）
PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT = PROJECT_ROOT / "datas"
RESULT_DIR = PROJECT_ROOT / "result" / "CNN_result" / "ifuku_train_test_rgb_black_mask_left40"

# データ一覧CSV
NAMELIST_PATH = DATA_ROOT / "record0-10xyz" / "namelist.csv"


# =========================================================
# 共通
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "weights").mkdir(exist_ok=True)
    (RESULT_DIR / "csv").mkdir(exist_ok=True)
    (RESULT_DIR / "preview").mkdir(exist_ok=True)


def build_input_image(bgr_img: np.ndarray, mode: str = "rgb") -> np.ndarray:
    if mode == "rgb":
        return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    elif mode == "g":
        g = bgr_img[:, :, 1]
        g3 = cv2.merge([g, g, g])
        return g3

    elif mode == "hs":
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        z = np.zeros_like(h, dtype=np.uint8)
        bgr_like = cv2.merge([z, s, h])
        return bgr_like

    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_mask(img: np.ndarray) -> np.ndarray:
    out = img.copy()

    if MASK_LEFT_PIXELS > 0:
        out[:, :MASK_LEFT_PIXELS, :] = 0
    if MASK_RIGHT_PIXELS > 0:
        out[:, -MASK_RIGHT_PIXELS:, :] = 0
    if MASK_TOP_PIXELS > 0:
        out[:MASK_TOP_PIXELS, :, :] = 0
    if MASK_BOTTOM_PIXELS > 0:
        out[-MASK_BOTTOM_PIXELS:, :, :] = 0

    return out


def normalize_y(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype("float32")
    y[:, 0] /= NORMAL_FORCE_NORMALIZE
    y[:, 1] += SHEAR_FORCE_NORMALIZE
    y[:, 2] += SHEAR_FORCE_NORMALIZE
    y[:, 1] /= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 2] /= (SHEAR_FORCE_NORMALIZE * 2.0)
    return y


def unnormalize_y(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype("float32")
    y[:, 0] *= NORMAL_FORCE_NORMALIZE
    y[:, 1] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 2] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 1] -= SHEAR_FORCE_NORMALIZE
    y[:, 2] -= SHEAR_FORCE_NORMALIZE
    return y


def resolve_img_path(path: str) -> Path:
    img_path = Path(path)
    if not img_path.is_absolute():
        path_str = str(img_path).replace("./", "", 1).replace(".\\", "", 1)
        img_path = PROJECT_ROOT / path_str
    return img_path


# =========================================================
# データ読み込み
# =========================================================
def parse_ifuku_id_from_path(path_str: str) -> int | None:
    parts = Path(path_str).parts
    for p in parts:
        if p.startswith("ifuku"):
            num_str = p.replace("ifuku", "")
            if num_str.isdigit():
                return int(num_str)
    return None


class DataLoader:
    def __init__(self, name: str = "ifuku", img_mode: str = "rgb"):
        self.name = name
        self.img_mode = img_mode

    def personal_dataload(self) -> pd.DataFrame:
        names = pd.read_csv(NAMELIST_PATH, header=None)

        all_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy", "ifuku_id"])

        for _, names_item in names.iterrows():
            relative_csv_path = str(names_item[0])
            csv_path = PROJECT_ROOT / "datas" / relative_csv_path

            if self.name not in relative_csv_path:
                continue

            if not csv_path.exists():
                print(f"[WARN] not found: {csv_path}")
                continue

            csv_record = pd.read_csv(csv_path, header=0)
            csv_record.columns = ["img_path", "Fz", "Fx", "Fy"]

            csv_record = csv_record[
                pd.to_numeric(csv_record["Fz"], errors="coerce") <= 100
            ].copy()

            csv_record["ifuku_id"] = csv_record["img_path"].apply(parse_ifuku_id_from_path)
            all_df = pd.concat([all_df, csv_record], ignore_index=True)

        all_df = all_df.dropna(subset=["ifuku_id"]).copy()
        all_df["ifuku_id"] = all_df["ifuku_id"].astype(int)

        return all_df

    def load_one_image(self, path: str) -> np.ndarray:
        img_path = resolve_img_path(path)

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"image not found: {img_path}")

        x = build_input_image(bgr, mode=self.img_mode)

        if x.shape[0] != IMG_H or x.shape[1] != IMG_W:
            x = cv2.resize(x, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        x = apply_mask(x)
        return x

    def df_to_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_list = []
        for p in df["img_path"].tolist():
            x_list.append(self.load_one_image(p))

        x = np.array(x_list, dtype="uint8")
        y = df[["Fz", "Fx", "Fy"]].values.astype("float32")

        x = x.astype("float32") / 255.0
        y_norm = normalize_y(y)

        return x, y, y_norm


class ImageDataSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, loader, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        x_list = []
        for p in batch_df["img_path"].tolist():
            x_list.append(self.loader.load_one_image(p))

        x = np.array(x_list, dtype="float32") / 255.0

        y_true = batch_df[["Fz", "Fx", "Fy"]].values.astype("float32")
        y_norm = normalize_y(y_true)

        return x, [y_norm[:, 0], y_norm[:, 1], y_norm[:, 2]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# モデル
# =========================================================
class MultiTaskCNN:
    def __init__(self, img_mode: str = "rgb"):
        self.img_mode = img_mode
        self.model_dir = RESULT_DIR
        self.model = self.build_model()

    def build_model(self) -> Model:
        l2_alpha = 0.001
        middle_class_recurrence = 342
        last_activation = "linear"

        optimizer = Adam(learning_rate=1e-4)

        input_tensor = Input(shape=(IMG_H, IMG_W, IMG_C), name="input_tensor")

        conv = VGG16(
            weights="imagenet",
            input_shape=(IMG_H, IMG_W, IMG_C),
            include_top=False
        )(input_tensor)

        flatten = GlobalMaxPooling2D(name="flatten")(conv)

        fz = Dense(middle_class_recurrence, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        fz = Dropout(0.2)(fz)
        fz = Dense(1, activation=last_activation, name="Fz",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fz)

        fx = Dense(middle_class_recurrence, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        fx = Dropout(0.2)(fx)
        fx = Dense(1, activation=last_activation, name="Fx",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fx)

        fy = Dense(middle_class_recurrence, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        fy = Dropout(0.2)(fy)
        fy = Dense(1, activation=last_activation, name="Fy",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fy)

        model = Model(input_tensor, [fz, fx, fy])
        model.compile(
            loss={"Fz": "mean_squared_error",
                  "Fx": "mean_squared_error",
                  "Fy": "mean_squared_error"},
            optimizer=optimizer
        )
        return model

    def save_architecture(self) -> None:
        model_path = self.model_dir / "for0-10.json"
        fig_path = self.model_dir / "for0-10.png"

        with open(model_path, "w", encoding="utf-8") as f:
            f.write(self.model.to_json())

        if not fig_path.exists():
            plot_model(self.model, show_shapes=True, to_file=str(fig_path))


# =========================================================
# 可視化
# =========================================================
def preview_before_after(df: pd.DataFrame, loader: DataLoader, save_dir: Path, n_samples: int = 6) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        return

    n_samples = min(n_samples, len(df))
    sample_df = df.sample(n=n_samples, random_state=RANDOM_SEED).reset_index(drop=True)

    plt.figure(figsize=(12, 4 * n_samples))

    for i in range(n_samples):
        img_path = sample_df.loc[i, "img_path"]
        img_path_obj = resolve_img_path(img_path)

        bgr = cv2.imread(str(img_path_obj))
        if bgr is None:
            continue

        before = build_input_image(bgr, mode=loader.img_mode)
        if before.shape[0] != IMG_H or before.shape[1] != IMG_W:
            before = cv2.resize(before, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        after = apply_mask(before)

        if loader.img_mode == "rgb":
            before_show = before
            after_show = after
        else:
            before_show = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
            after_show = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.imshow(before_show)
        plt.title(f"Before\nifuku={sample_df.loc[i, 'ifuku_id']}")
        plt.axis("off")

        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.imshow(after_show)
        plt.title("After black mask")
        plt.axis("off")

        out_path = save_dir / f"sample_{i+1:02d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(after_show, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    grid_path = save_dir / "before_after_preview.png"
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] preview saved -> {grid_path}")


# =========================================================
# 評価
# =========================================================
def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    out = {}
    names = ["Fz", "Fx", "Fy"]

    for i, name in enumerate(names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        out[f"{prefix}_{name}_MAE"] = mae
        out[f"{prefix}_{name}_RMSE"] = rmse

    return out


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    ensure_dirs()

    loader = DataLoader(name=SUBJECT_NAME, img_mode=IMG_MODE)
    full_df = loader.personal_dataload()

    print("=== full data loaded ===")
    print(full_df[["img_path", "ifuku_id"]].head())

    trainval_df = full_df[
        (full_df["ifuku_id"] >= TRAINVAL_START) & (full_df["ifuku_id"] <= TRAINVAL_END)
    ].copy()

    test_df = full_df[
        (full_df["ifuku_id"] >= TEST_START) & (full_df["ifuku_id"] <= TEST_END)
    ].copy()

    trainval_ids = list(range(TRAINVAL_START, TRAINVAL_END + 1))
    random.Random(RANDOM_SEED).shuffle(trainval_ids)

    val_size = max(1, int(len(trainval_ids) * VAL_RATIO))
    val_ids = sorted(trainval_ids[:val_size])
    train_ids = sorted(trainval_ids[val_size:])

    train_df = trainval_df[trainval_df["ifuku_id"].isin(train_ids)].reset_index(drop=True)
    val_df = trainval_df[trainval_df["ifuku_id"].isin(val_ids)].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("=== split ===")
    print("train ifuku:", train_ids)
    print("val   ifuku:", val_ids)
    print("test  ifuku:", list(range(TEST_START, TEST_END + 1)))

    print(f"train samples: {len(train_df)}")
    print(f"val samples  : {len(val_df)}")
    print(f"test samples : {len(test_df)}")

    print("=== preview masked images ===")
    preview_before_after(train_df, loader, RESULT_DIR / "preview" / "train", n_samples=6)
    preview_before_after(val_df, loader, RESULT_DIR / "preview" / "val", n_samples=4)
    preview_before_after(test_df, loader, RESULT_DIR / "preview" / "test", n_samples=4)

    print("=== build train/val sequence ===")
    train_seq = ImageDataSequence(train_df, loader, batch_size=BATCH_SIZE, shuffle=True)
    val_seq = ImageDataSequence(val_df, loader, batch_size=BATCH_SIZE, shuffle=False)

    print("=== build model ===")
    net = MultiTaskCNN(img_mode=IMG_MODE)
    net.save_architecture()
    net.model.summary()

    weight_path = RESULT_DIR / "weights" / f"weight_{SUBJECT_NAME}_for0-10.h5"

    if weight_path.exists():
        print("=== saved weights found: skip training ===")
        print(weight_path)
        net.model.load_weights(str(weight_path))
    else:
        print("=== train ===")
        history = net.model.fit(
            train_seq,
            epochs=EPOCHS,
            validation_data=val_seq,
            verbose=1
        )

        net.model.save_weights(str(weight_path))

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(
            RESULT_DIR / "csv" / "learning_log.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("=== load test images only ===")
    x_test, y_test_true, y_test_norm = loader.df_to_xy(test_df)

    print("=== predict test ===")
    test_pred_list = net.model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    y_test_pred = unnormalize_y(np.concatenate(test_pred_list, axis=1))

    print("=== evaluate ===")
    metrics = {}
    metrics.update(calc_metrics(y_test_true, y_test_pred, "test"))

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        RESULT_DIR / "csv" / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_result_df = test_df.copy()
    test_result_df["pred_Fz"] = y_test_pred[:, 0]
    test_result_df["pred_Fx"] = y_test_pred[:, 1]
    test_result_df["pred_Fy"] = y_test_pred[:, 2]

    test_result_df["err_Fz"] = test_result_df["pred_Fz"] - test_result_df["Fz"]
    test_result_df["err_Fx"] = test_result_df["pred_Fx"] - test_result_df["Fx"]
    test_result_df["err_Fy"] = test_result_df["pred_Fy"] - test_result_df["Fy"]

    test_result_df.to_csv(
        RESULT_DIR / "csv" / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_group_df = (
        test_result_df.groupby("ifuku_id")[["err_Fz", "err_Fx", "err_Fy"]]
        .mean()
        .reset_index()
    )
    test_group_df.to_csv(
        RESULT_DIR / "csv" / "test_group_mean_error.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=== done ===")
    print(metrics_df.T)


if __name__ == "__main__":
    main()