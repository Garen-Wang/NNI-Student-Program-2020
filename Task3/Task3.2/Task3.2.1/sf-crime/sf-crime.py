import sys
sys.path.append('../../')
import warnings
warnings.filterwarnings('ignore')
import logging
import nni
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from fe_util import name2feature
from model import get_fea_importance

logger = logging.getLogger('sf-crime')
train_file_name = '../data/sf-crime/train.csv'
test_file_name = '../data/sf-crime/test.csv'
# train_file_name = 'sf-crime-train.csv'
# test_file_name = 'sf-crime-test.csv'
target_name = 'Category'
target_columns = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY',
       'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE',
       'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
       'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
       'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING',
       'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES',
       'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE',
       'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
       'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE',
       'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
       'WARRANTS', 'WEAPON LAWS']
id_index = 'Id'
train = pd.read_csv(train_file_name)
test = pd.read_csv(test_file_name)


def fillna():
    global train, test, target_columns, id_index
    test_id = test[id_index]
    train.drop(['Descript', 'Resolution'], axis=1, inplace=True)
    test.drop(id_index, axis=1, inplace=True)
    if train[target_name].dtypes == object:
        target_encoder = LabelEncoder()
        train[target_name] = target_encoder.fit_transform(train[target_name])
        target_columns = target_encoder.classes_
    train['Dates'] = train['Dates'].map(lambda x: pd.to_datetime(x))
    train['Year'] = train['Dates'].map(lambda x: x.year)
    train['Month'] = train['Dates'].map(lambda x: x.month)
    train['Day'] = train['Dates'].map(lambda x: x.day)
    train['Hour'] = train['Dates'].map(lambda x: x.hour)

    test['Dates'] = test['Dates'].map(lambda x: pd.to_datetime(x))
    test['Year'] = test['Dates'].map(lambda x: x.year)
    test['Month'] = test['Dates'].map(lambda x: x.month)
    test['Day'] = test['Dates'].map(lambda x: x.day)
    test['Hour'] = test['Dates'].map(lambda x: x.hour)
    dayofweek_dict = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    pd_district_encoder = LabelEncoder()
    pd_district_encoder.fit(train['PdDistrict'])
    train['DayOfWeek'] = train['DayOfWeek'].map(lambda x: dayofweek_dict[x])
    train['PdDistrict'] = pd_district_encoder.transform(train['PdDistrict'])
    train['Quarter'] = train['Dates'].map(lambda x: x.quarter)

    test['DayOfWeek'] = test['DayOfWeek'].map(lambda x: dayofweek_dict[x])
    test['PdDistrict'] = pd_district_encoder.transform(test['PdDistrict'])
    test['Quarter'] = test['Dates'].map(lambda x: x.quarter)

    def getHourZone(x):
        if x <= 7:
            return 1
        elif x <= 11:
            return 2
        elif x <= 13:
            return 3
        elif x <= 16:
            return 4
        elif x <= 18:
            return 5
        else:
            return 6

    train['HourZone'] = train['Hour'].map(lambda x: getHourZone(x))
    # data[data['Y'] == 90]
    train['HasAddressNum'] = train['Address'].map(lambda x: 1 if 'Block' in x else 0)
    # train['AddressNum'] = train['Address'].map(lambda x: x.split(' ')[0])

    test['HourZone'] = test['Hour'].map(lambda x: getHourZone(x))
    # data[data['Y'] == 90]
    test['HasAddressNum'] = test['Address'].map(lambda x: 1 if 'Block' in x else 0)
    # test['AddressNum'] = test['Address'].map(lambda x: x.split(' ')[0])

    # def isint(x):
    #     try:
    #         int(x)
    #         return True
    #     except ValueError:
    #         pass
    #     return False

    # train['AddressNum'] = train['AddressNum'].map(lambda x: int(x) if isint(x) else -1)
    # test['AddressNum'] = test['AddressNum'].map(lambda x: int(x) if isint(x) else -1)
    # address_nums = train['AddressNum'].unique()
    # for address_num in address_nums:
    #     if address_num == -1:
    #         train['AddressNum-' + 'None'] = train['AddressNum'].map(lambda x: 1 if x == -1 else 0)
    #         test['AddressNum-' + 'None'] = test['AddressNum'].map(lambda x: 1 if x == -1 else 0)
    #     else:
    #         train['AddressNum-' + str(address_num)] = train['AddressNum'].map(lambda x: 1 if x == address_num else 0)
    #         test['AddressNum-' + str(address_num)] = test['AddressNum'].map(lambda x: 1 if x == address_num else 0)
    # train.drop('AddressNum', axis=1, inplace=True)
    # test.drop('AddressNum', axis=1, inplace=True)
    def getLocation(x):
        if 'Block' in x:
            locations.add(x.split('of')[-1].strip())
            return [x.split('of')[-1].strip()]
        else:
            locations.add(x.split('/')[0].strip())
            locations.add(x.split('/')[1].strip())
            return [x.split('/')[0].strip(), x.split('/')[1].strip()]
    locations = set()
    test['Locations'] = test['Address'].apply(getLocation)
    locations = set()
    train['Locations'] = train['Address'].apply(getLocation)
    locations = list(locations)
    for location in locations:
        if len(location.split(' ')[-1]) != 2:
            locations.remove(location)
    suffixes = set()
    for location in locations:
        suffixes.add(location.split(' ')[-1])
    suffixes = list(suffixes)

    def hasSuffix(li, suffix):
        for x in li:
            if x.split(' ')[-1] == suffix:
                return 1
        return 0
    for suffix in suffixes:
        train['suffix-' + suffix] = train['Locations'].map(lambda x: hasSuffix(x, suffix))
        test['suffix-' + suffix] = test['Locations'].map(lambda x: hasSuffix(x, suffix))

    def hasOthers(li):
        for x in li:
            if x.split(' ')[-1] not in suffixes:
                return 1
        return 0
    train['suffix-OTHERS'] = train['Locations'].map(hasOthers)
    test['suffix-OTHERS'] = test['Locations'].map(hasOthers)

    train.drop(['Dates', 'Locations', 'Address'], axis=1, inplace=True)
    test.drop(['Dates', 'Locations', 'Address'], axis=1, inplace=True)
    return test_id


def lightgbm_train(max_epoch=1000, min_data=200):
    # min_data: 1, 3, 9, 97561, 292683
    global train, target_name
    params_lgb = {
        'task': 'train',
        'objective': 'multiclass',
        'num_class': 39,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        # 'min_data': min_data,
        'max_depth': 128
    }
    X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Category', axis=1), train['Category'].values, test_size=0.15, random_state=2077)
    # print(X_train)
    # print(y_train)
    # print(X_valid)
    # print(y_valid)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    classifier = lgb.train(params=params_lgb, train_set=lgb_train, valid_sets=lgb_valid, verbose_eval=50, valid_names='eval', num_boost_round=max_epoch)
    feature_names = [feature_name for feature_name in train.columns if feature_name != target_name]
    # feature_importance = get_fea_importance(classifier, feature_names)
    from sklearn.metrics import log_loss
    y_pred = classifier.predict(X_valid, num_iteration=classifier.best_iteration)
    print(y_pred)
    return classifier


def predict(classifier):
    global test, target_columns
    pred = classifier.predict(test, num_iteration=classifier.best_iteration)
    for i in range(pred.shape[0]):
        pred[i, np.argmax(pred[i])] = 1.
    pred = pred.astype(int)
    df = pd.DataFrame(pred)
    df.to_csv('sf-crime-prediction.csv', index=False)


def main():
    global train, test
    nni_params = nni.get_next_parameter()
    # logger.info("Received params:\n", nni_params)
    if 'sample_feature' in nni_params.keys():
        sample_col = nni_params['sample_feature']
    else:
        sample_col = []
    train = name2feature(train, sample_col, target_name)

    classifier = lightgbm_train(min_data=9)
    predict(classifier)


if __name__ == '__main__':
    test_id = fillna()
    train.to_csv('sf-crime-train.csv', index=False)
    test.to_csv('sf-crime-test.csv', index=False)
    test_id.to_csv('sf-crime-test-id.csv', index=False)
