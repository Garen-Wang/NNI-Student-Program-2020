import logging
import numpy as np
import pandas as pd
import nni
from sklearn import preprocessing


logger = logging.getLogger('auto-feature-engineering')

if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'

    params = nni.get_next_parameter()
    logger.info("received info: {}".format(params))

    data = pd.read_csv(file_name)
    if 'sample_feature' in params.keys():
        sample_col = params['sample_feature']
    else:
        sample_col = []

    feature = get_feature(data, sample_col, target_name)

    feature_importance, val_score = lgb_model_train()

    nni.report_final_result({
        'default': val_score,
        'feature_importance': feature_importance
    })
