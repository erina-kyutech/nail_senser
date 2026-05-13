import os
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Model

SUBJECT_NAME = "ifuku"
MODEL_DIR = r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_rgb_concat_150x290"
IMAGE_DIR = r"C:\Users\Owner\PycharmProjects\datas\record0-10xyz\ifuku1\360deg"
DATALOG_CSV = os.path.join(IMAGE_DIR, "datalog.csv")
OUTPUT_DIR = r"C:\Users\Owner\PycharmProjects\result\gradcam_rgb_10images"

TARGET_IMAGE_COUNT = 10
RANDOM_SEED = None
TARGETS = ["Fz", "Fx", "Fy"]

LEFT_WIDTH = 150
RIGHT_WIDTH = 140


def load_rgb_model(model_dir: str, subject_name: str):
    model_json_path = os.path.join(model_dir, "for0-10.json")
    weight_path = os.path.join(model_dir, "weight", f"weight_{subject_name}_for0-10.h5")

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"model json not found: {model_json_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight not found: {weight_path}")

    print("MODEL_DIR   :", model_dir)
    print("MODEL_JSON  :", model_json_path)
    print("WEIGHT_PATH :", weight_path)
    print("JSON_MTIME  :", os.path.getmtime(model_json_path))
    print("WEIGHT_MTIME:", os.path.getmtime(weight_path))

    with open(model_json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string)
    model.load_weights(weight_path)
    return model


def list_numbered_images(image_dir: str):
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(".png"):
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                files.append((int(stem), os.path.join(image_dir, name)))
    files.sort(key=lambda x: x[0])
    if not files:
        raise FileNotFoundError(f"No numbered png images found in: {image_dir}")
    return files


def pick_images():
    numbered = list_numbered_images(IMAGE_DIR)
    count = min(TARGET_IMAGE_COUNT, len(numbered))
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    selected = random.sample(numbered, count)
    selected.sort(key=lambda x: x[0])
    return selected


def load_datalog():
    if not os.path.exists(DATALOG_CSV):
        raise FileNotFoundError(f"datalog.csv not found: {DATALOG_CSV}")

    df = pd.read_csv(DATALOG_CSV, header=None)
    first_cell = str(df.iloc[0, 0]).lower()
    if not first_cell.endswith(".png"):
        df = pd.read_csv(DATALOG_CSV)

    df = df.iloc[:, :4].copy()
    df.columns = ["path", "Fz", "Fr", "Ff"]
    return df


def load_force_from_datalog(df, image_index: int):
    target_name = f"{image_index}.png"
    matched = df[df["path"].astype(str).str.endswith(target_name)]
    if matched.empty:
        raise ValueError(f"No row found in datalog.csv for image: {target_name}")

    row = matched.iloc[0]
    return {
        "Fz_true": float(row["Fz"]),
        "Fr_true": float(row["Fr"]),
        "Ff_true": float(row["Ff"]),
        "path_in_csv": str(row["path"]),
    }


def load_input_rgb(image_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"image not found: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return rgb, x


def build_feature_extractor(model):
    """
    外側グラフ上に存在する flatten 層の入力を特徴マップとして使う。
    これは VGG16 の最後の特徴マップ (block5_pool 出力相当) なので、
    nested model の graph disconnected を避けられる。
    """
    flatten_layer = model.get_layer("flatten")
    feature_model = Model(inputs=model.inputs, outputs=flatten_layer.input)
    print("gradcam_feature_tensor_from:", "flatten.input")
    return feature_model


def make_gradcam_heatmap(model, img_array, target_name):
    target_index_map = {"Fz": 0, "Fx": 1, "Fy": 2}
    target_idx = target_index_map[target_name]

    flatten_layer = model.get_layer("flatten")

    grad_model = Model(
        inputs=model.inputs,
        outputs=[flatten_layer.input, model.outputs[target_idx]]
    )

    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        feature_maps, preds = grad_model(img_tensor, training=False)
        target_score = preds[:, 0]

    grads = tape.gradient(target_score, feature_maps)
    if grads is None:
        raise RuntimeError(f"Gradient が計算できませんでした: {target_name}")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    max_val = tf.reduce_max(heatmap)
    if float(max_val.numpy()) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def overlay_heatmap_on_rgb(rgb_img, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_color_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb_img, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    return heatmap_resized, overlay


def save_rgb(path, rgb_img):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def calc_left_right_ratio(heatmap_resized):
    left_sum = float(np.sum(heatmap_resized[:, :LEFT_WIDTH]))
    right_sum = float(np.sum(heatmap_resized[:, LEFT_WIDTH:LEFT_WIDTH + RIGHT_WIDTH]))
    total = left_sum + right_sum
    if total == 0:
        return 0.0, 0.0
    return left_sum / total, right_sum / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datalog_df = load_datalog()
    selected_images = pick_images()

    print("selected image count:", len(selected_images))
    print("selected indexes:", [idx for idx, _ in selected_images])

    model = load_rgb_model(MODEL_DIR, SUBJECT_NAME)
    feature_model = build_feature_extractor(model)

    summary_rows = []
    selected_rows = []

    for image_index, image_path in selected_images:
        print("=" * 60)
        print("image_index:", image_index)
        print("image_path :", image_path)

        force_info = load_force_from_datalog(datalog_df, image_index)
        rgb_img, x = load_input_rgb(image_path)

        preds = model(x, training=False)
        preds_np = [float(p.numpy()[0, 0]) for p in preds]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_subdir = os.path.join(OUTPUT_DIR, f"img_{base_name}")
        os.makedirs(image_subdir, exist_ok=True)

        original_path = os.path.join(image_subdir, f"{base_name}_original.png")
        save_rgb(original_path, rgb_img)

        selected_rows.append({
            "image_index": image_index,
            "image_path": image_path,
            "path_in_csv": force_info["path_in_csv"],
            "true_Fz": force_info["Fz_true"],
            "true_Fr": force_info["Fr_true"],
            "true_Ff": force_info["Ff_true"],
            "pred_Fz": preds_np[0],
            "pred_Fx": preds_np[1],
            "pred_Fy": preds_np[2],
        })

        for target_name in TARGETS:
            heatmap = make_gradcam_heatmap(model, x, target_name)
            heatmap_resized, overlay = overlay_heatmap_on_rgb(rgb_img, heatmap, alpha=0.4)

            raw_heatmap_path = os.path.join(image_subdir, f"{base_name}_{target_name}_heatmap.png")
            overlay_path = os.path.join(image_subdir, f"{base_name}_{target_name}_overlay.png")

            cv2.imwrite(raw_heatmap_path, heatmap_resized)
            save_rgb(overlay_path, overlay)

            left_ratio, right_ratio = calc_left_right_ratio(heatmap_resized)

            summary_rows.append({
                "image_index": image_index,
                "image_path": image_path,
                "path_in_csv": force_info["path_in_csv"],
                "target": target_name,
                "true_Fz": force_info["Fz_true"],
                "true_Fr": force_info["Fr_true"],
                "true_Ff": force_info["Ff_true"],
                "pred_Fz": preds_np[0],
                "pred_Fx": preds_np[1],
                "pred_Fy": preds_np[2],
                "left_ratio": left_ratio,
                "right_ratio": right_ratio,
                "original_path": original_path,
                "raw_heatmap_path": raw_heatmap_path,
                "overlay_path": overlay_path,
            })

            print(f"[{target_name}] left_ratio={left_ratio:.3f}, right_ratio={right_ratio:.3f}")

    selected_df = pd.DataFrame(selected_rows)
    summary_df = pd.DataFrame(summary_rows)

    selected_csv = os.path.join(OUTPUT_DIR, "selected_images_summary.csv")
    summary_csv = os.path.join(OUTPUT_DIR, "gradcam_summary_all.csv")

    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("saved:", selected_csv)
    print("saved:", summary_csv)
    print("done")


if __name__ == "__main__":
    main()
