from __future__ import print_function
import os
import gc

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K, regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model

tf.compat.v1.disable_eager_execution()


def directry_initialize():
    """ソースコードのあるフォルダから1つ上のディレクトリへ移動する。"""
    nowdir = os.path.dirname(__file__)
    os.chdir(nowdir)
    os.chdir("..")


def build_input_image(bgr_img):
    """
    BGR画像をRGB画像へ変換して返す。
    RGBのみを学習に使うので、他モードの分岐は削除。
    """
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


class data_loader(object):
    def __init__(self, name=None, Fz_range=10.0, dummy_flag=False, num_workers=8):
        self.dummy_flag = dummy_flag
        self.name = name

        if Fz_range == 5.0:
            self.normal_force_normalize = 5.0
            self.thear_force_normalize = 4.0
        elif Fz_range == 10.0:
            self.normal_force_normalize = 10.0
            self.thear_force_normalize = 5.0
        else:
            raise ValueError(f"Unsupported Fz_range: {Fz_range}")

    def personal_dataload(self):
        if self.dummy_flag:
            namelist_path = "./datas/record0-10xyz/namelist_dum.csv"
        else:
            namelist_path = "./datas/record0-10xyz/namelist.csv"

        names = pd.read_csv(namelist_path, header=None)
        alldatas_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy"])

        for _, names_item in names.iterrows():
            now_namedir = "./datas/" + names_item[0]

            if self.name in now_namedir:
                csv_record = pd.read_csv(now_namedir, header=0)
                csv_record.columns = ["path", "Fz", "Fr", "Ff"]

                csv_record = csv_record[pd.to_numeric(csv_record["Fz"], errors="coerce") <= 100].copy()

                append_df = pd.concat(
                    [csv_record["path"], csv_record["Fz"], csv_record["Fr"], csv_record["Ff"]],
                    axis=1
                )
                append_df.columns = ["img_path", "Fz", "Fx", "Fy"]
                alldatas_df = pd.concat([alldatas_df, append_df], ignore_index=True)

        return alldatas_df

    def data_normalize(self, X, Y):
        X = X.astype("float32").reshape((-1, 150, 290, 3))
        X /= 255.0

        Y = Y.astype("float32").copy()
        Y[:, 0] /= self.normal_force_normalize
        Y[:, 1] += self.thear_force_normalize
        Y[:, 2] += self.thear_force_normalize
        Y[:, 1] /= (self.thear_force_normalize * 2)
        Y[:, 2] /= (self.thear_force_normalize * 2)
        return X, Y

    def data_unnormalize(self, Y):
        Y = Y.astype("float32").copy()
        Y[:, 0] *= self.normal_force_normalize
        Y[:, 1] *= (self.thear_force_normalize * 2)
        Y[:, 2] *= (self.thear_force_normalize * 2)
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize
        return Y


class multitask_CNN(object):
    def __init__(self):
        self.model_dir = "./result/CNN_result/vgg16_rgb_concat_150x290/"

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "weight"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "indiv_score"), exist_ok=True)

        self.CNN_build()

    def CNN_build(self):
        l2_alpha = 0.001
        middle_class_recurrence = 342
        last_activation = "linear"
        optimizer = Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        model_path = os.path.join(self.model_dir, "for0-10.json")
        model_fig_path = os.path.join(self.model_dir, "for0-10.png")

        input_tensor = Input(shape=(150, 290, 3), name="input_tensor")

        conv = VGG16(
            weights="imagenet",
            input_shape=(150, 290, 3),
            include_top=False
        )(input_tensor)

        flatten = GlobalMaxPooling2D(name="flatten")(conv)

        Fz = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(flatten)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(
            1,
            activation=last_activation,
            name="Fz",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(Fz)

        Fx = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(flatten)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(
            1,
            activation=last_activation,
            name="Fx",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(Fx)

        Fy = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(flatten)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(
            1,
            activation=last_activation,
            name="Fy",
            kernel_regularizer=regularizers.l2(l2_alpha)
        )(Fy)

        self.model = Model(input_tensor, [Fz, Fx, Fy])

        self.model.compile(
            loss={
                "Fz": "mean_squared_error",
                "Fx": "mean_squared_error",
                "Fy": "mean_squared_error"
            },
            optimizer=optimizer
        )

        print("MODEL_DIR:", self.model_dir)
        print("MODEL_INPUT:", self.model.input_shape)

        if not os.path.exists(model_path):
            json_string = self.model.to_json()
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(json_string)

        if not os.path.exists(model_fig_path):
            plot_model(self.model, show_shapes=True, to_file=model_fig_path)

    def model_load_from_path(self, name):
        model_json_path = os.path.join(self.model_dir, "for0-10.json")
        with open(model_json_path, "r", encoding="utf-8") as f:
            model_json_string = f.read()

        self.model = model_from_json(
            model_json_string,
            custom_objects={"tf": tf, "K": K}
        )

        model_weight_path = os.path.join(self.model_dir, "weight", f"weight_{name}_for0-10.h5")
        self.model.load_weights(model_weight_path)


class Trainer(object):
    def __init__(self, model_obj, datas_obj):
        self.datas = datas_obj
        self.model = model_obj
        self.name = self.datas.name

        os.makedirs(os.path.join(self.model.model_dir, "indiv_score", self.name), exist_ok=True)

        self.epoch = 2 if self.datas.dummy_flag else 10
        self.batch_size = 32
        self._debug_saved = False

    def base_train(self):
        callbacks = []
        personaldatas_df = self.datas.personal_dataload()

        data_length = len(personaldatas_df)
        index_array = np.random.permutation(np.arange(data_length, dtype="uint32"))

        train_rate = 0.85
        train_index_array, test_index_array = np.split(
            index_array,
            [int(len(index_array) * train_rate)]
        )

        X_train, Y_train = self.data_indexread(personaldatas_df, train_index_array)
        X_test, Y_test = self.data_indexread(personaldatas_df, test_index_array)

        X_train, Y_train = self.datas.data_normalize(X_train, Y_train)
        X_test, Y_test = self.datas.data_normalize(X_test, Y_test)

        history = self.model.model.fit(
            X_train,
            [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
            epochs=self.epoch,
            validation_data=(X_test, [Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]]),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        if self.datas.dummy_flag:
            weight_name = f"weight_{self.name}_for0-10_dum.h5"
            log_name = "learning_log_dum.csv"
        else:
            weight_name = f"weight_{self.name}_for0-10.h5"
            log_name = "learning_log.csv"

        weight_path = os.path.join(self.model.model_dir, "weight", weight_name)
        self.model.model.save_weights(weight_path)

        log_csvpath = os.path.join(self.model.model_dir, "indiv_score", self.name, log_name)
        log_df = pd.DataFrame(
            history.history,
            columns=[
                "Fx_loss", "Fy_loss", "Fz_loss", "loss",
                "val_Fx_loss", "val_Fy_loss", "val_Fz_loss", "val_loss"
            ]
        )
        log_df.to_csv(log_csvpath, index=False)

        self.evaluate_save(personaldatas_df, train_index_array, keyword="train")
        self.evaluate_save(personaldatas_df, test_index_array, keyword="val")

    def data_indexread(self, datas_df, index_array):
        def path2img(path):
            print("\r", "now image loading", end="")
            bgr = cv2.imread(path)
            if bgr is None:
                raise FileNotFoundError(f"image not found: {path}")
            return build_input_image(bgr)

        index_list = list(index_array)
        X_path_list = list(datas_df.iloc[index_list, 0])
        X_img_list = list(map(path2img, X_path_list))

        if not self._debug_saved:
            self._debug_saved = True
            dbg_dir = f"./debug_input/{self.datas.name}/rgb"
            os.makedirs(dbg_dir, exist_ok=True)

            for i, img in enumerate(X_img_list[:16]):
                out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(dbg_dir, f"sample_{i:02d}.png"), out_bgr)

            print(f"[DEBUG] saved samples -> {dbg_dir}")

        X_array = np.array(X_img_list)
        Y_array = datas_df.iloc[index_list, [1, 2, 3]].values
        return X_array, Y_array

    def evaluate_save(self, personaldatas_df, index_array, keyword="train", chunk_size=1024, pred_batch_size=128):
        save_dir = os.path.join(self.model.model_dir, "indiv_score", self.name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"evaluate_{keyword}_for0-10.csv")

        rows = []
        N = len(index_array)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_idx = index_array[start:end]

            X, Y = self.data_indexread(personaldatas_df, batch_idx)
            X, Y = self.datas.data_normalize(X, Y)

            pred = self.model.model.predict(X, batch_size=pred_batch_size, verbose=0)

            if isinstance(pred, (list, tuple)) and len(pred) == 3:
                pred_Fz, pred_Fx, pred_Fy = pred
            else:
                pred_Fz, pred_Fx, pred_Fy = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

            pred_concat = np.concatenate([pred_Fz, pred_Fx, pred_Fy], axis=1)
            pred_concat = self.datas.data_unnormalize(pred_concat)
            true_concat = self.datas.data_unnormalize(Y.copy())

            for i in range(len(batch_idx)):
                rows.append({
                    "Fz_predict": float(pred_concat[i, 0]),
                    "Fx_predict": float(pred_concat[i, 1]),
                    "Fy_predict": float(pred_concat[i, 2]),
                    "Fz_true": float(true_concat[i, 0]),
                    "Fx_true": float(true_concat[i, 1]),
                    "Fy_true": float(true_concat[i, 2]),
                    "Fz_error": float(pred_concat[i, 0] - true_concat[i, 0]),
                    "Fx_error": float(pred_concat[i, 1] - true_concat[i, 1]),
                    "Fy_error": float(pred_concat[i, 2] - true_concat[i, 2]),
                })

            print(f"[{keyword}] evaluated {end}/{N} samples")
            del X, Y, pred, pred_concat, true_concat
            gc.collect()

        pd.DataFrame(rows).to_csv(save_path, index=False)
        print("saved:", save_path)


if __name__ == "__main__":
    dummy_flag = False

    if dummy_flag:
        namelist = ["ifuku"]
    else:
        namelist = ["ifuku"]

    directry_initialize()

    for now_name in namelist:
        print(f"\n===== TRAIN: name={now_name}, mode=rgb =====")

        CNN = multitask_CNN()
        database = data_loader(
            name=now_name,
            Fz_range=10.0,
            dummy_flag=dummy_flag
        )
        trainer = Trainer(CNN, database)
        trainer.base_train()

        del CNN, database, trainer
        gc.collect()
