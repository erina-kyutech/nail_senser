from __future__ import print_function
from re import X
from statistics import mean
import os
from collections import deque
import numpy as np
import tensorflow as tf
import cv2
from keras.applications import VGG16
from tensorflow import keras
from keras import regularizers
from keras import backend as K
from keras.models import Model,model_from_json
from keras.layers import Input, Dense, Dropout, Lambda

from tensorflow.python.keras import models
from keras.optimizers import adam_v2 as Adam
from tensorflow.python.keras.constraints import non_neg
from keras.layers import GlobalMaxPooling2D

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import smtplib

#ディレクトリの移動
#データセットのあるフォルダや結果を保存するフォルダへアクセスするために
#ソースコードのあるフォルダから1つ上のディレクトリへ移動
def directry_initialize():
    nowdir = os.path.dirname(__file__)  #プログラムのあるディレクトリを参照
    os.chdir(nowdir)  #作業ディレクトリをプログラムのあるディレクトリに
    os.chdir("..")#一つ上のディレクトリに移動

#データを読み出すためのクラス
class data_loader(object):
    def __init__(self,name=None,Fz_range=10.0,dummy_flag=False,img_mode="rgb",num_workers=8):
        self.dummy_flag=dummy_flag
        self.name=name
        self.img_mode = img_mode
        #垂直力の測定範囲に応じて正規化定数を変える
        if Fz_range==5.0:
            self.normal_force_normalize = 5.0
            self.thear_force_normalize = 4.0

        elif Fz_range==10.0:
            self.normal_force_normalize = 10.0
            self.thear_force_normalize = 5.0

    #個人のデータを読む
    def personal_dataload(self):
        #ダミーモードの時はデータのごく一部しか読み込まない
        #csvパスの指定
        if self.dummy_flag:
            namelist_path = "./datas/record0-10xyz/namelist_dum.csv"
        else:
            namelist_path = "./datas/record0-10xyz/namelist.csv"

        #ネームリストを読み込み
        names = pd.read_csv(namelist_path, header=None)

        alldatas_df =pd.DataFrame(columns=["img_path","Fz","Fx","Fy"])

        #ネームリスト内のパスを順に参照しその中のデータを順に格納
        for  names_index,names_item in names.iterrows():
            #各フォルダのデータリスト(画像のパスと測定した力を格納したcsv)のパスを参照
            now_namedir ="./datas/" +  names_item[0]

            #引数の文字列(被験者名)がパスに含まれているものだけを読み込み
            if self.name in now_namedir:
                #画像パスと指先力測定値を記録したcsvを開く
                csv_record = pd.read_csv(now_namedir, header=0)

                now_degree = int(names_item[1])     #せん断角度[degree]
                now_rad = np.deg2rad(now_degree)    #せん断角度[radian]

                csv_record.columns=["path","Fz","Fr","Ff"]
                img_path = csv_record["path"]
                Fz = csv_record["Fz"]
                Fx = csv_record["Fr"]
                Fy = csv_record["Ff"]

                append_df=pd.concat([img_path,Fz,Fx,Fy],axis=1)
                append_df.columns=["img_path","Fz","Fx","Fy"]
                alldatas_df=pd.concat([alldatas_df,append_df])
    
        return alldatas_df


    #データの正規化
    def data_normalize(self,X,Y):

        #Xの正規化
        X = X.astype("float32")
        #(データ数,画像縦長さ,画像横長さ,色数)となるよう次元を調整
        # reshape depending on mode
        if self.img_mode == "g":
            X = X.reshape((-1,150,290,1))
        else:
            X = X.reshape((-1,150,290,3))

        #255で割って0~1.0の範囲にする
        X/=255.0

        #Yの正規化
        #垂直力の正規化(0~1の範囲に)
        Y[:,0] /= self.normal_force_normalize
        #せんだん力の正規化(0~1の範囲に)
        Y[:,1] += self.thear_force_normalize
        Y[:,2] += self.thear_force_normalize
        Y[:,1] /= (self.thear_force_normalize*2)
        Y[:,2] /= (self.thear_force_normalize*2)
        return X,Y

    #正規化されたデータを元に戻す関数
    def data_unnormalize(self,Y):
        #垂直力を戻す
        Y[:,0] *= self.normal_force_normalize
        #せんだん力の正規化
        Y[:,1] *= (self.thear_force_normalize*2)
        Y[:,2] *= (self.thear_force_normalize*2)
        Y[:,1] -= self.thear_force_normalize
        Y[:,2] -= self.thear_force_normalize

        return Y

#自作層(グレースケールをカラー画像にする)関数
def tensor_gray2BGR(grayX):
    blank = tf.zeros_like(grayX)
    BGR_X = tf.concat([blank,grayX],axis=3)
    BGR_X = tf.concat([BGR_X,blank],axis=3)
    
    return BGR_X


class multitask_CNN(object):
    #クラスを呼び出したときに同時に呼び出される関数
    def __init__(self):
        self.model_dir = "./result/CNN_result/vgg16_rgb_concat_150x290/"
        os.makedirs(self.model_dir+"weight", exist_ok=True)
        os.makedirs(self.model_dir+"indiv_score", exist_ok=True)
        self.CNN_build()

    def load_from_dir(self, model_dir, subject="ifuku"):
        self.model_dir = model_dir if model_dir.endswith(os.sep) else model_dir + os.sep
        json_path = os.path.join(self.model_dir, "for0-10.json")
        weight_path = os.path.join(self.model_dir, "weight", f"weight_{subject}_for0-10.h5")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"json not found: {json_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"weight not found: {weight_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            model_json_string = f.read()
        self.model = model_from_json(model_json_string, custom_objects={"tf": tf, "K": K})
        self.model.load_weights(weight_path)
        return self.model

    def CNN_build(self):
        l2_alpha = 0.001
        middle_class_recurrence = 342
        last_activation = "linear"

        input_tensor = Input(shape=(150,290,3),name="input_tensor")
        conv = VGG16(weights="imagenet",
                     input_shape=(150,290,3),
                     include_top=False)(input_tensor)
        flatten = GlobalMaxPooling2D(name="flatten")(conv)

        Fz = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(1,
                    activation=last_activation,
                    name="Fz",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fz)

        Fx = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(1,
                    activation=last_activation,
                    name="Fx",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fx)

        Fy = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(1,
                    activation=last_activation,
                    name="Fy",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fy)

        predicts =[Fz,Fx,Fy]
        self.model = Model(input_tensor,predicts)

    def model_load_from_path(self, subject="ifuku"):

        json_path = os.path.join(self.model_dir, "for0-10.json")
        weight_path = os.path.join(self.model_dir, "weight", f"weight_{subject}_for0-10.h5")

        with open(json_path, "r", encoding="utf-8") as f:
            model_json_string = f.read()

        self.model = model_from_json(model_json_string, custom_objects={"tf": tf, "K": K})
        self.model.load_weights(weight_path)
        return self.model


#実際に学習に使うクラス
#引数にCNNクラス、データ読み出しクラスをとる
class Trainer(object):
    def __init__(self,model_obj,datas_obj):
        self.datas=datas_obj
        self.model=model_obj

        self.name=self.datas.name



    #一人のデータ
    def base_train(self):
        #個人データを読み取る
        personaldatas_df = self.datas.personal_dataload()

        #データを並べ替え(X,Yの相関は保ったまま)
        data_length=len(personaldatas_df)
        index_array=np.array(range(data_length),dtype="uint32")
        index_array = np.random.permutation(index_array) 

        #評価用データ
        X_test,Y_test=self.data_indexread(personaldatas_df,index_array)

        #正規化
        X_test,Y_test = self.datas.data_normalize(X_test,Y_test)

        #=============================ここから評価データでの評価==================================================
        self.evaluate_save(personaldatas_df,index_array,keyword="val")


    #データの読み込み
    #mapを使うことでfor文より早く読み出せる
    #Xが入力画像
    #Yがそれに対応する指先力
    def data_indexread(self,datas_df,index_array):
        #map用関数
        def path2img(path):
            print("\r","now image loading",end="")
            #パスから画像そのものを読み出してリストに格納
            # img_mode: 'g' (grayscale), 'rgb', 'hs'
            if getattr(self.datas, "img_mode", "g") == "g":
                return cv2.imread(path, 0)
            else:
                bgr = cv2.imread(path, 1)
                if bgr is None:
                    return None
                mode = getattr(self.datas, "img_mode", "rgb")
                if mode == "rgb":
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif mode == "hs":
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    h = hsv[:, :, 0]
                    s = hsv[:, :, 1]
                    z = np.zeros_like(h)
                    return np.stack([h, s, z], axis=2)
                else:
                    return bgr

        #データを読み出し
        index_list=list(index_array)
        X_path_list=list(datas_df.iloc[index_list,0])
        X_img_list=list(map(path2img,X_path_list,))

        #np.arrayに変換
        X_array=np.array(X_img_list)
        # drop None (failed reads)
        if X_array.dtype == object:
            X_img_list2 = [x for x in X_img_list if x is not None]
            if len(X_img_list2) != len(X_img_list):
                keep_mask = [x is not None for x in X_img_list]
                index_list = [k for k, keep in zip(index_list, keep_mask) if keep]
            X_array = np.array(X_img_list2)

        if self.datas.img_mode == "g":
            X_array=X_array.reshape(-1,150,290,1)
        else:
            X_array=X_array.reshape(-1,150,290,3)
        Y_array=datas_df.iloc[index_list,[1,2,3]].values
        return X_array,Y_array

    #推定値と真値との差を記録する関数、keywordで名前を設定
    def evaluate_save(self,datas_df,eval_index_array,keyword=""):
        #データ格納用のディレクトリがあるか確認、無かったら作る
        indivisual_dir = self.model.model_dir+"indiv_score/"+self.name+"/"
        if os.path.exists(indivisual_dir):
            pass
        else:
            os.makedirs(indivisual_dir, exist_ok=True)

        #データを分割して読み出す数
        split_num = 30000
        #データの総数
        data_length=len(eval_index_array)
        index_split_num=int(data_length/split_num)+1

        #分割データ数がデータ総数より多いなら一括処理
        if  split_num > data_length:
            #データを読み出し
            X,Y=self.data_indexread(datas_df,eval_index_array)
         
            #正規化
            X,Y=self.datas.data_normalize(X,Y)
            #評価する
            Y_predict_list=self.model.model.predict(X)
            Y_predict=np.concatenate([Y_predict_list[0],
                                      Y_predict_list[1],
                                      Y_predict_list[2]],
                                     axis=1)

            #正規化状態から戻す
            Y_true=self.datas.data_unnormalize(Y)
            Y_predict=self.datas.data_unnormalize(Y_predict)

            Fz_pred= Y_predict[:,0]
            Fx_pred= Y_predict[:,1]
            Fy_pred= Y_predict[:,2]
            Fz_true= Y_true[:,0]
            Fx_true= Y_true[:,1]
            Fy_true= Y_true[:,2]
            Fz_error= Fz_pred - Fz_true
            Fx_error= Fx_pred - Fx_true
            Fy_error= Fy_pred - Fy_true

            eval_log_df = pd.DataFrame([
                                        Fz_pred,
                                        Fx_pred,
                                        Fy_pred,
                                        Fz_true,
                                        Fx_true,
                                        Fy_true,
                                        Fz_error,
                                        Fx_error,
                                        Fy_error],
                                        index=[
                                        "Fz_predict",
                                        "Fx_predict",
                                        "Fy_predict",
                                        "Fz_true",
                                        "Fx_true",
                                        "Fy_true",
                                        "Fz_error",
                                        "Fx_error",
                                        "Fy_error"])
        #分割データ数がデータ総数より少ないなら分割処理
        #そうしないとGPUのメモリが足りないので
        else:
            index_array = np.array_split(eval_index_array,index_split_num)


            #分割数分くりかえす
            read_length=len(index_array)
            for i in range(read_length):
                now_index_array = index_array[i]

                #データを読み出し
                X,Y=self.data_indexread(datas_df,now_index_array)

                X,Y=self.datas.data_normalize(X,Y)
                #評価する
                Y_predict_list=self.model.model.predict(X,
                                                        batch_size=128,
                                                        verbose=1)
                Y_predict=np.concatenate([Y_predict_list[0],
                                          Y_predict_list[1],
                                          Y_predict_list[2]],
                                         axis=1)
                #正規化状態から戻す
                Y_predict= self.datas.data_unnormalize(Y_predict)
                Y_true=self.datas.data_unnormalize(Y)

                Fz_pred= Y_predict[:,0]
                Fx_pred= Y_predict[:,1]
                Fy_pred= Y_predict[:,2]
                Fz_true= Y_true[:,0]
                Fx_true= Y_true[:,1]
                Fy_true= Y_true[:,2]
                Fz_error= Fz_pred - Fz_true
                Fx_error= Fx_pred - Fx_true
                Fy_error= Fy_pred - Fy_true

                #評価データをDataFrameに格納
                if i==0:
                    eval_log_df = pd.DataFrame([
                                                    Fz_pred,
                                                    Fx_pred,
                                                    Fy_pred,
                                                    Fz_true,
                                                    Fx_true,
                                                    Fy_true,
                                                    Fz_error,
                                                    Fx_error,
                                                    Fy_error],
                                                    index=[
                                                    "Fz_predict",
                                                    "Fx_predict",
                                                    "Fy_predict",
                                                    "Fz_test",
                                                    "Fx_test",
                                                    "Fy_test",
                                                    "Fz_error",
                                                    "Fx_error",
                                                    "Fy_error"])
                else:
                    concat_df = pd.DataFrame([
                                                    Fz_pred,
                                                    Fx_pred,
                                                    Fy_pred,
                                                    Fz_true,
                                                    Fx_true,
                                                    Fy_true,
                                                    Fz_error,
                                                    Fx_error,
                                                    Fy_error],
                                                    index=[
                                                    "Fz_predict",
                                                    "Fx_predict",
                                                    "Fy_predict",
                                                    "Fz_test",
                                                    "Fx_test",
                                                    "Fy_test",
                                                    "Fz_error",
                                                    "Fx_error",
                                                    "Fy_error"])
                    eval_log_df = pd.concat([eval_log_df,concat_df],axis=1)
        if self.datas.dummy_flag:
            eval_log_path=indivisual_dir + "evaluate_"+keyword+"_for0-10_dum.csv"
        else:
            eval_log_path=indivisual_dir + "evaluate_"+keyword+"_for0-10.csv"
        eval_log_df = eval_log_df.transpose()

        #DataFrameをcsv形式で保存
        eval_log_df.to_csv(eval_log_path,encoding="shift-jis")

    # ============================
    # Real-time用：3モデル同時ロード
    # ============================
def load_model_for_mode(mode: str, subject: str = "ifuku", base_dir: str = None):
    """
    mode: "rgb" / "g" / "hs"
    期待フォルダ（例）:
      C:\\Users\\Owner\\PycharmProjects\\result\\CNN_result\\vgg16_rgb_concat_150x290\\for0-10.json
      C:\\Users\\Owner\\PycharmProjects\\result\\CNN_result\\vgg16_rgb_concat_150x290\\weight\\weight_ifuku_for0-10.h5
    """
    mode = mode.lower().strip()
    if mode not in ("rgb", "g", "hs"):
        raise ValueError(f"mode must be one of rgb/g/hs, got: {mode}")

    if base_dir is None:
        base_dir = r"C:\Users\Owner\PycharmProjects\result\CNN_result"

    folder_map = {
        "rgb": "vgg16_rgb_concat_150x290",
        "g": "vgg16_g_concat_150x290",
        "hs": "vgg16_hs_concat_150x290",
    }

    model_dir = os.path.join(base_dir, folder_map[mode])
    json_path = os.path.join(model_dir, "for0-10.json")
    weight_path = os.path.join(model_dir, "weight", f"weight_{subject}_for0-10.h5")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"json not found: {json_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight not found: {weight_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string, custom_objects={"tf": tf, "K": K})
    model.load_weights(weight_path)
    return model

def load_models_all(subject: str = "ifuku", base_dir: str = None):
    """
    rgb/g/hs をまとめて返す
    """
    return {
        "rgb": load_model_for_mode("rgb", subject=subject, base_dir=base_dir),
        "g": load_model_for_mode("g", subject=subject, base_dir=base_dir),
        "hs": load_model_for_mode("hs", subject=subject, base_dir=base_dir),
    }


if __name__ == "__main__":
    # ===== settings =====
    namelist = ["ifuku"]  # add more subjects if needed

    # This file defines a grayscale-based model (input shape (H,W,1) + gray2BGR).
    # If your trained model is rgb/hs, edit multitask_CNN.CNN_build accordingly.
    modes = ["rgb"]

    for now_name in namelist:
        for mode in modes:
            print(f"\n===== VALIDATION: name={now_name}, mode={mode}, size=150x290 =====")
            CNN = multitask_CNN()

            # load trained model structure
            CNN.model_load_from_path()

            # load trained weights (expects weight/weight_<name>_for0-10.h5 under CNN.model_dir)
            weight_path = CNN.model_dir + f"weight/weight_{now_name}_for0-10.h5"
            if os.path.exists(weight_path):
                CNN.model.load_weights(weight_path)
            else:
                print("WARNING: weight file not found:", weight_path)

            database = data_loader(name=now_name, Fz_range=10.0, dummy_flag=False, img_mode=mode)
            trainer = Trainer(CNN, database)
            trainer.base_train()
