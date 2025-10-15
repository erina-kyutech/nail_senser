import subprocess
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

# 実行したい.pyファイルのリスト
script_list = ["simple_learning_VGG16.py", "simple_learning_VGG19.py", "simple_learning_ResNet50.py", "simple_learning_Xception.py"]

for script in script_list:
    # サブスクリプトファイルを順に実行
    subprocess.run(["python", script])