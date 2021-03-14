import nni
import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from fe_util import *
from model import *


if __name__ == '__main__':
    file_name = './heart.csv'
    target_name = 'label'
    id_index = 'id'
    min_data = 10

    params = nni.get_next_parameter()
    
    df = pd.read_csv(file_name, sep=' ')
    df.columns = [
        "age", "sex", "chest-pain", "bp-resting", "cholesterol", "bs-fasting", "ecg-resting", "hr-max", "eia", "oldpeak", "k-oldpeak", "vessels", "thal", "label"
    ]
    df['label'] -= 1
    
    if 'sample_feature' in params.keys():
        sample_col = params['sample_feature']
    else:
        sample_col = []

    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df, _epoch=1000, target_name=target_name, id_index=id_index, min_data=min_data)
    nni.report_final_result({
        "default": val_score,
        "feature_importance": feature_imp
    })

    print(feature_imp)
    print(val_score)

