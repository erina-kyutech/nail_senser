# -*- coding: utf-8 -*-
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model


SUBJECT_NAME = "ifuku"
MODEL_DIR = r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_rgb_concat_150x290"


def main():
    json_path = os.path.join(MODEL_DIR, "for0-10.json")
    weight_path = os.path.join(MODEL_DIR, "weight", f"weight_{SUBJECT_NAME}_for0-10.h5")
    fig_path = os.path.join(MODEL_DIR, "model_structure_check.png")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"json not found: {json_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight not found: {weight_path}")

    print("=== 読み込むモデル ===")
    print("MODEL_DIR  :", MODEL_DIR)
    print("JSON_PATH  :", json_path)
    print("WEIGHT_PATH:", weight_path)
    print()

    # モデル構造を復元
    with open(json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string)
    model.load_weights(weight_path)

    print("=== model.summary() ===")
    model.summary()

    print("\n=== 入出力確認 ===")
    print("model.input_shape :", model.input_shape)
    print("model.output_shape:", model.output_shape)
    print("output names      :", model.output_names)

    print("\n=== 各層の名前と出力shape ===")
    for i, layer in enumerate(model.layers):
        try:
            print(f"{i:2d}  {layer.name:20s}  output_shape={layer.output_shape}")
        except Exception:
            print(f"{i:2d}  {layer.name:20s}  output_shape=取得不可")

    # flatten層確認
    print("\n=== flatten層の確認 ===")
    flatten_layer = model.get_layer("flatten")
    print("layer class   :", flatten_layer.__class__.__name__)
    print("layer name    :", flatten_layer.name)
    print("flatten.input :", flatten_layer.input.shape)
    print("flatten.output:", flatten_layer.output.shape)

    # モデル図保存（graphviz/pydot が入っていれば）
    try:
        plot_model(model, to_file=fig_path, show_shapes=True, show_layer_names=True)
        print(f"\nモデル図を保存しました: {fig_path}")
    except Exception as e:
        print("\nplot_model は保存できませんでした。")
        print("理由:", e)


if __name__ == "__main__":
    main()