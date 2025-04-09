import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate
from scipy.fft import fft
from scipy.signal import find_peaks
from lazypredict.Supervised import LazyClassifier




data_train_file_1   = pd.read_csv(r"F:\Python_Dataset\Mechine Learning\Dataset\BUN_DO\Gen_data\File_1.csv")
data_train_file_4   = pd.read_csv(r"F:\Python_Dataset\Mechine Learning\Dataset\BUN_DO\Gen_data\File_4.csv")

data_test_file_2    = pd.read_csv(r"F:\Python_Dataset\Mechine Learning\Dataset\BUN_DO\Gen_data\File_2.csv")


data_train = pd.concat([data_train_file_1, data_train_file_4], ignore_index=True)


data_train = data_train.drop(['Unnamed: 0', 'action'], axis = 1)
data_train['operation'] = data_train['operation'].astype('category')

data_test_file_2 = data_test_file_2.drop(['Unnamed: 0', 'action'], axis = 1)
data_test_file_2['operation'] = data_test_file_2['operation'].astype('category')




def calculate_elbow_angle(data, upper_arm_sensor, wrist_sensor):
    ua = data[[f'{upper_arm_sensor}/acc_x', f'{upper_arm_sensor}/acc_y', f'{upper_arm_sensor}/acc_z']].to_numpy()
    wf = data[[f'{wrist_sensor}/acc_x', f'{wrist_sensor}/acc_y', f'{wrist_sensor}/acc_z']].to_numpy()
    vec_forearm = wf - ua
    dot = np.einsum('ij,ij->i', ua, vec_forearm)
    norm = np.linalg.norm(ua, axis=1) * np.linalg.norm(vec_forearm, axis=1) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))

def calculate_cross_corr_lag(data, s1, s2, axis='acc_z'):
    x = data[f'{s1}/{axis}'].values
    y = data[f'{s2}/{axis}'].values
    corr = correlate(x - x.mean(), y - y.mean(), mode='full')
    lag = np.argmax(corr) - (len(y) - 1)
    return lag

def feature_extraction(data):
    for i in range(1, 5):
        data[f'atr0{i}/acc_mag'] = np.sqrt(
            data[f'atr0{i}/acc_x']**2 + data[f'atr0{i}/acc_y']**2 + data[f'atr0{i}/acc_z']**2)
        data[f'atr0{i}/gyro_mag'] = np.sqrt(
            data[f'atr0{i}/gyro_x']**2 + data[f'atr0{i}/gyro_y']**2 + data[f'atr0{i}/gyro_z']**2)

    for i in range(1, 5):
        quat = data[[f'atr0{i}/quat_x', f'atr0{i}/quat_y', f'atr0{i}/quat_z', f'atr0{i}/quat_w']].to_numpy()
        rot = R.from_quat(quat)
        angles = np.linalg.norm(rot.as_rotvec(), axis=1)
        data[f'atr0{i}/rot_angle_deg'] = np.degrees(angles)

    for i in [1, 2]:
        quat = data[[f'atr0{i}/quat_x', f'atr0{i}/quat_y', f'atr0{i}/quat_z', f'atr0{i}/quat_w']].to_numpy()
        r = R.from_quat(quat)
        euler = r.as_euler('xyz', degrees=True)
        data[f'atr0{i}/roll'] = euler[:, 0]
        data[f'atr0{i}/pitch'] = euler[:, 1]
        data[f'atr0{i}/yaw'] = euler[:, 2]

    for i in [3, 4]:
        z = data[f'atr0{i}/acc_z']
        norm = np.sqrt(
            data[f'atr0{i}/acc_x']**2 + data[f'atr0{i}/acc_y']**2 + data[f'atr0{i}/acc_z']**2) + 1e-8
        data[f'atr0{i}/arm_elevation'] = np.degrees(np.arccos(np.clip(z / norm, -1, 1)))

    data['acc_mag_diff_01_02'] = data['atr01/acc_mag'] - data['atr02/acc_mag']
    data['acc_mag_diff_03_04'] = data['atr03/acc_mag'] - data['atr04/acc_mag']
    data['acc_mag_diff_01_03'] = data['atr01/acc_mag'] - data['atr03/acc_mag']
    data['acc_mag_diff_02_04'] = data['atr02/acc_mag'] - data['atr04/acc_mag']

    data['elbow_angle_right'] = calculate_elbow_angle(data, 'atr03', 'atr01')
    data['elbow_angle_left'] = calculate_elbow_angle(data, 'atr04', 'atr02')

    data['lag_right_arm'] = calculate_cross_corr_lag(data, 'atr03', 'atr01', 'acc_z')
    data['lag_left_arm'] = calculate_cross_corr_lag(data, 'atr04', 'atr02', 'acc_z')

    for i in [1, 2, 3, 4]:
        sensor = f'atr0{i}'
        fft_vals = np.abs(fft(data[f'{sensor}/acc_x'].values))
        data[f'{sensor}/dominant_freq'] = np.argmax(fft_vals[:len(fft_vals)//2])

    for i in [1, 2]:
        data[f'atr0{i}/wrist_stability'] = data[f'atr0{i}/roll'].rolling(window=10, min_periods=1).std()

    data['pseudo_torque_right'] = (data['atr03/gyro_mag'] - data['atr01/gyro_mag']) * (data['atr03/acc_mag'] + data['atr01/acc_mag']) * 0.3
    data['pseudo_torque_left'] = (data['atr04/gyro_mag'] - data['atr02/gyro_mag']) * (data['atr04/acc_mag'] + data['atr02/acc_mag']) * 0.3

    data['wrist_angular_speed'] = data['atr01/gyro_mag'].rolling(window=5, min_periods=1).max()
    data['upper_arm_acc_before_throw'] = data['atr03/acc_mag'].shift(10).rolling(window=10, min_periods=1).mean()
    data['elbow_angle_stable'] = data['elbow_angle_right'].rolling(window=15, min_periods=1).std() < 5

    # Bổ sung đặc trưng nâng cao
    for i in range(1, 5):
        data[f'atr0{i}/jerk'] = data[f'atr0{i}/acc_mag'].diff().fillna(0) * 30

    for i in [1, 2]:
        data[f'atr0{i}/stillness'] = (data[f'atr0{i}/gyro_mag'] < 5).rolling(window=15, min_periods=1).sum()

    for i in [1, 2]:
        data[f'atr0{i}/roll_range'] = data[f'atr0{i}/roll'].rolling(window=30).apply(lambda x: x.max() - x.min(), raw=True)

    for i in [1, 2, 3, 4]:
        data[f'atr0{i}/acc_energy'] = data[f'atr0{i}/acc_mag'].rolling(window=30).sum()

    data['elbow_velocity'] = data['elbow_angle_right'].diff().fillna(0) * 30

    from scipy.stats import entropy
    for i in [1, 2]:
        data[f'atr0{i}/acc_entropy'] = data[f'atr0{i}/acc_mag'].rolling(window=30).apply(
            lambda x: entropy(np.histogram(x, bins=10, density=True)[0] + 1e-6), raw=True)

    data['corr_acc_lr'] = data['atr01/acc_mag'].rolling(30).corr(data['atr02/acc_mag'])

    for i in [3, 4]:
        data[f'atr0{i}/vertical_orientation'] = data[f'atr0{i}/acc_z'] / (data[f'atr0{i}/acc_mag'] + 1e-6)

    return data


data_train          = feature_extraction(data_train)
data_test_file_2    = feature_extraction(data_test_file_2)




print("Kích thước dữ liệu train:", data_train.shape)
print("Kích thước dữ liệu test:", data_test_file_2.shape)

data_train.dropna(inplace=True)
data_test_file_2.dropna(inplace=True)


print("Kích thước dữ liệu train:", data_train.shape)
print("Kích thước dữ liệu test:", data_test_file_2.shape)

target = "operation"

X_train = data_train.drop(target, axis=1)
y_train = data_train[target]

X_test = data_test_file_2.drop(target, axis=1)
y_test = data_test_file_2[target]

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Trước SMOTE:", pd.Series(y_train_encoded).value_counts())
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)
print("Sau SMOTE:", pd.Series(y_train_resampled).value_counts())

print("Nhãn trong y_train_resampled:", np.unique(y_train_resampled))
print("Nhãn trong y_test_encoded:", np.unique(y_test_encoded))


clf = LazyClassifier(verbose=3, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_resampled, X_test, y_train_resampled, y_test_encoded)

print(models)


#
#                                Accuracy  ...  Time Taken
# Model                                    ...
# RandomForestClassifier             0.58  ...      401.23
# ExtraTreesClassifier               0.58  ...       93.83
# LGBMClassifier                     0.58  ...       42.57
# XGBClassifier                      0.58  ...       53.23
# NuSVC                              0.55  ...    12840.11
# SVC                                0.57  ...     5368.02
# LinearDiscriminantAnalysis         0.50  ...       16.05
# LogisticRegression                 0.51  ...       18.14
# CalibratedClassifierCV             0.48  ...      393.38
# LinearSVC                          0.48  ...      117.71
# SGDClassifier                      0.46  ...       22.18
# GaussianNB                         0.48  ...        2.16
# BaggingClassifier                  0.49  ...      481.00
# NearestCentroid                    0.45  ...        1.65
# RidgeClassifier                    0.43  ...        1.69
# RidgeClassifierCV                  0.43  ...        6.44
# AdaBoostClassifier                 0.41  ...      235.75
# KNeighborsClassifier               0.37  ...       36.35
# BernoulliNB                        0.38  ...        2.04
# DecisionTreeClassifier             0.40  ...       69.98
# Perceptron                         0.38  ...       10.35
# PassiveAggressiveClassifier        0.36  ...        8.82
# QuadraticDiscriminantAnalysis      0.33  ...        4.02
# ExtraTreeClassifier                0.31  ...        1.92
# DummyClassifier                    0.10  ...        1.06