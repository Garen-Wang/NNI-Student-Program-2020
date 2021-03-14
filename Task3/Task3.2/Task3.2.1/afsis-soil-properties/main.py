import nni
import numpy as np
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from lightgbm import LGBMRegressor

import sys
sys.path.append('../../')

from fe_util import *
from model import *

targets = ['Ca', 'P', 'pH', 'SOC', 'Sand']

def lgb_train(train, test, targets, epoch=1000, id_col='id', min_data=100):
    # print(train[targets])
    # X_train, X_val, y_train, y_val = train_test_split(df.drop(exclusive, axis=1), df[targets], random_state=42, test_size=0.25)
    y_train = train[targets]
    exclusive = targets.copy()
    exclusive.append(id_col)
    X_train = train.drop(exclusive, axis=1)
    X_test = test.drop(id_col, axis=1)
    regressor = MultiOutputRegressor(LGBMRegressor(random_state=42, n_jobs=-1)).fit(X_train, y_train)
    pred = regressor.predict(X_test)
    feature_importance = np.zeros((3594,))
    for i in range(len(regressor.estimators_)):
        feature_importance += regressor.estimators_[i].feature_importances_
    feature_importance /= len(regressor.estimators_)
    df = pd.DataFrame(pred, columns=targets)
    df.insert(0, id_col, test[id_col])
    df.to_csv('baseline.csv', index=False)
    return 1, feature_importance



if __name__ == '__main__':
    train = pd.read_csv('../../data/afsis-soil-properties/training.csv')
    train['Depth'] = train['Depth'].map(lambda x: 1 if x == 'Topsoil' else 0)
    params = nni.get_next_parameter()
    sample_col = params['sample_feature'] if 'sample_feature' in params.keys() else []
    train = name2feature(train, sample_col)
    test = pd.read_csv('../../data/afsis-soil-properties/sorted_test.csv')
    test['Depth'] = test['Depth'].map(lambda x: 1 if x == 'Topsoil' else 0)
    # feature_importance, val_auc = lgb_train(train, targets, epoch=500, id_col='PIDN', min_data=13) # 89
    val_auc, feature_importance = lgb_train(train, test, targets=targets, epoch=500, id_col='PIDN', min_data=13) # 89
    nni.report_final_result({
        'default': val_auc,
        'feature_importance': feature_importance
    })
    print(feature_importance)
    
