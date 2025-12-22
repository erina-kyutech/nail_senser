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
    def __init__(self,name=None,Fz_range=10.0,dummy_flag=False,img_mode="g",num_workers=8):
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
            namelist_path = "./datas/record0-10xyz/namelist_all10.csv"

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

                csv_record.columns=["path","Fz","Fr"]
                img_path = csv_record["path"]                
                Fz = csv_record["Fz"]               
                Fx = csv_record["Fr"]* np.cos(now_rad)
                Fy = csv_record["Fr"]* np.sin(now_rad)

                append_df=pd.concat([img_path,Fz,Fx,Fy],axis=1)
                append_df.columns=["img_path","Fz","Fx","Fy"]
                alldatas_df=pd.concat([alldatas_df,append_df])
    
        return alldatas_df


    #データの正規化
    def data_normalize(self,X,Y):

        #Xの正規化
        X = X.astype("float64")
        #(データ数,画像縦長さ,画像横長さ,色数)となるよう次元を調整
        X = X.reshape((-1,155,140,1))#グレースケールなので色数1

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
        #モデル構造や重み、結果を保存するディレクトリ
        self.model_dir = "C:/Users"+"/Owner"+"/PycharmProjects/result/CNN_result/vgg16_gray"+ "/"

        #指定したディレクトリがなければ作る
        if os.path.exists(self.model_dir):
            pass
        else:
            os.mkdir(self.model_dir)
            os.mkdir(self.model_dir+"weight")
            os.mkdir(self.model_dir+"indiv_score")

        #モデルを構築
        self.CNN_build()

    #モデルの構築
    def CNN_build(self):
        l2_alpha = 0.001    #L2正則化の係数
        he_normal_seed = 111    #乱数シード

        #隠れ層ノード数は(入力ノード数+出力ノード数)*2/3が目安らしい
        middle_class_recurrence = 342   #全結合中間層のノード数
        
        last_activation = "linear"  #最終層の活性化関数

        #最適化手法
        #optimizer=Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        #モデルの保存するパス


        #ここからモデルの構築
        #構築方法はFunctionAPI
        #入力画像(画像縦サイズ、横サイズ、チャンネル数を指定)
        #縦155pixel,横140pixelのグレースケール画像(チャンネル数1)を入力
        input_tensor = Input(shape=(155,140,1),name="input_tensor")

        #VGG16の構造を読み込む(注:入力するのがカラー画像でないと学習済み重み値を利用できない)
        #VGG16で抽出した特徴量をFlattenで1次元化
        #グレースケール画像を赤、青要素0のカラー画像(チャンネル数3)に変換
        conv_input = Lambda(tensor_gray2BGR,
                            output_shape=(155,140,3),
                            name="gray2BGR")(input_tensor)
        conv_inputshape = (155,140,3)

        #VGG16を呼び出し(全結合層は含まない)
        conv = VGG16(weights="imagenet",
                     input_shape=conv_inputshape,
                     include_top=False)(conv_input)

        #畳み込み層の出力を1次元化
        flatten = GlobalMaxPooling2D(name="flatten")(conv)
    
        ##ここから全結合層を構築
        #Multitask-CNNなのでタスクごとに全結合層を作る
        #それぞれ活性化関数と正則化を指定している。
        #過学習防止にDropout層をつけている

        #垂直力推定タスク(回帰)
        Fz = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(1,
                    activation=last_activation,
                    name="Fz",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fz)

        #せん断力推定タスク(回帰)
        Fx = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(1,
                    activation=last_activation,
                    name="Fx",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fx)

        #せん断力推定タスク(回帰)
        Fy = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(1,
                    activation=last_activation,
                    name="Fy",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Fy)

        # 角度回帰タスク(回帰)
        Angle = Dense(middle_class_recurrence,
                    activation='relu',
                    kernel_regularizer = regularizers.l2(l2_alpha))(flatten)
        Angle = Dropout(0.2)(Angle)
        Angle = Dense(1,
                    activation=last_activation,
                    name="Angle",
                    kernel_regularizer = regularizers.l2(l2_alpha))(Angle)

        #最終出力のリストを作成
        predicts =[Fz,Fx,Fy,Angle]
            
        #モデルを構成
        self.model = Model(input_tensor,predicts)
        model_json_path =  self.model_dir + "for0-10.json"
        model_json_string = open(model_json_path).read()

        #モデル構造の読み込み
        self.model = model_from_json(model_json_string,
                                     custom_objects={'tf': tf,'K': K})
        model_weight_path = self.model_dir+"weight/weight_ifuku_for0-10.h5"
        self.model.load_weights(model_weight_path)

    #モデル構造と重みを読み込み
    def model_load_from_path(self):
        model_json_path =  self.model_dir + "for0-10.json"
        model_json_string = open(model_json_path).read()

        #モデル構造の読み込み
        self.model = model_from_json(model_json_string,
                                     custom_objects={'tf': tf,'K': K})


        #モデル重みの読み込み
        model_weight_path = self.model_dir+"/weight/weight_for0-10.h5"
        self.model.load_weights(model_weight_path)
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
            return cv2.imread(path,0)

        #データを読み出し
        index_list=list(index_array)
        X_path_list=list(datas_df.iloc[index_list,0])
        X_img_list=list(map(path2img,X_path_list,))

        #np.arrayに変換
        X_array=np.array(X_img_list)
        X_array=X_array.reshape(-1,155,140,1)
        Y_array=datas_df.iloc[index_list,[1,2,3]].values
        return X_array,Y_array

    #推定値と真値との差を記録する関数、keywordで名前を設定
    def evaluate_save(self,datas_df,eval_index_array,keyword=""):
        #データ格納用のディレクトリがあるか確認、無かったら作る
        indivisual_dir = self.model.model_dir+"indiv_score/"+self.name+"/"
        if os.path.exists(indivisual_dir):
            pass
        else:
            os.mkdir(indivisual_dir)

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

if __name__ == "__main__":

    #namelist = ["hamada","hanai","isogai","komura","mochiduki","sho","takeshige","tsuji","chen"]
    namelist = ["ifuku"]
    modes = ["rgb", "g", "hs"]
    #ディレクトリ移動(共通の処理)
    directry_initialize()

    #一気に学習
    for now_name in namelist:
        test_df = make_test_df(now_name)
        for mode in modes:
            model = load_model_for_mode(mode)
            y_true, y_pred = validate(model, test_df, img_mode=mode)
            save_results(y_true, y_pred, mode)

            # #モデル構築クラスの用意
            # CNN = multitask_CNN()
            #
            # #データ読み出しクラスの用意
            # database=data_loader(name=now_name,Fz_range=10.0,dummy_flag=False)
            #
            # #学習用クラスの用意
            # trainer = Trainer(CNN,database)
            #
            # #学習の実施
            # trainer.base_train()
            #
            # #次の学習に備えてオブジェクトの消去
            # del CNN,database,trainer
