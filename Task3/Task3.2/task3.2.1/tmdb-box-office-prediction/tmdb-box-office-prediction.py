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

from fe_util import *
train = pd.read_csv('../../data/tmdb-box-office-prediction/train.csv')
test = pd.read_csv('../../data/tmdb-box-office-prediction/test.csv')

target_name = 'revenue'

def fillna():
    global train, test
    rename_dict = {
        'belongs_to_collection': 'belongs-to-collection', 'imdb_id': 'imdb-id',
        'original_language': 'original-language', 'original_title': 'original-title',
        'poster_path': 'poster-path', 'production_companies': 'production-companies',
        'production_countries': 'production-countries', 'release_date': 'release-date',
        'spoken_languages': 'spoken-languages'
    }
    train.rename(columns=rename_dict, inplace=True)
    test.rename(columns=rename_dict, inplace=True)
    fillna_dict = {
        'belongs-to-collection': '[]', 'genres': '[]', 'homepage': "",
        'overview': "", 'production-companies': '[]', 'production-countries': '[]',
        'spoken-languages': '[]', 'Keywords': '[]', 'tagline': '',
        'cast': '[]', 'crew': '[]', 'status': '', 'title': ''
    }
    train.fillna(fillna_dict, inplace=True)
    train['runtime'].interpolate(method='values', inplace=True)
    test.fillna(fillna_dict, inplace=True)
    test['runtime'].interpolate(method='values', inplace=True)
    test.loc[828, 'release-date']: '5/1/00'

    train.drop(['poster-path', 'imdb-id'], axis=1, inplace=True)
    test.drop(['poster-path', 'imdb-id'], axis=1, inplace=True)

    # data['log-budget'] = np.log1p(data['budget'])
    train['budget'] = np.log1p(train['budget'])
    test['budget'] = np.log1p(test['budget'])
    train['has-homepage'] = train['homepage'].map(lambda x: 1 if x != '' else 0)
    test['has-homepage'] = test['homepage'].map(lambda x: 1 if x != '' else 0)

    train['release-date'] = train['release-date'].map(lambda x: pd.to_datetime(x))
    train['year'] = train['release-date'].map(lambda x: x.year)
    train['month'] = train['release-date'].map(lambda x: x.month)
    train['day'] = train['release-date'].map(lambda x: x.day)
    train['quarter'] = train['release-date'].map(lambda x: x.quarter)
    train['dayofweek'] = train['release.date'].map(lambda x: x.dayofweek + 1)

    test['release-date'] = test['release-date'].map(lambda x: pd.to_datetime(x))
    test['year'] = test['release-date'].map(lambda x: x.year)
    test['month'] = test['release-date'].map(lambda x: x.month)
    test['day'] = test['release-date'].map(lambda x: x.day)
    test['quarter'] = test['release-date'].map(lambda x: x.quarter)
    test['dayofweek'] = test['release.date'].map(lambda x: x.dayofweek + 1)

    train['genres-count'] = train['genres'].map(lambda x: len(eval(x)))
    test['genres-count'] = test['genres'].map(lambda x: len(eval(x)))
    train['companies-count'] = train['production-companies'].map(lambda x: len(eval(x)))
    test['companies-count'] = test['production-companies'].map(lambda x: len(eval(x)))
    train['countries-count'] = train['production-countries'].map(lambda x: len(eval(x)))
    test['countries-count'] = test['production-countries'].map(lambda x: len(eval(x)))
    train['cast-count'] = train['cast'].map(lambda x: len(eval(x)))
    test['cast-count'] = test['cast'].map(lambda x: len(eval(x)))
    train['crew-count'] = train['crew'].map(lambda x: len(eval(x)))
    test['crew-count'] = test['crew'].map(lambda x: len(eval(x)))

    def exist_name(val, name):
        for x in eval(val):
            if x['name'] == name:
                return 1
        return 0
    # collection begin
    train['has-collection'] = train['belongs-to-collection'].map(lambda x: 1 if x != [] else 0)
    test['has-collection'] = test['belongs-to-collection'].map(lambda x: 1 if x != [] else 0)
    collection_dict = {}
    for val in train['belongs-to-collection']:
        for v in eval(val):
            if v['name'] not in collection_dict.keys():
                collection_dict[v['name']] = 1
            else:
                collection_dict[v['name']] += 1
    good_collections = sorted(collection_dict.items(), key=lambda x: x[1], reverse=True)[:33]
    for name, _ in good_collections:
        train['collection-' + name] = train['belongs-to-collection'].map(
            lambda x: exist_name(x, name))
        test['collection-' + name] = test['belongs-to-collection'].map(
            lambda x: exist_name(x, name))
    # collection end

    # cast crew begin
    cast_dict = {}
    for val in train['cast']:
        for v in eval(val):
            if v['name'] not in cast_dict.keys():
                cast_dict[v['name']] = 1
            else:
                cast_dict[v['name']] += 1
    famous_actors = sorted(cast_dict.items(), key=lambda x: x[1], reverse=True)[:30]
    for famous_actor, _ in famous_actors:
        train['famous-cast-' + famous_actor] = train['cast'].map(lambda x: exist_name(x, famous_actor))
        test['famous-cast-' + famous_actor] = test['cast'].map(lambda x: exist_name(x, famous_actor))
    crew_dict = {}
    for val in train['crew']:
        for v in eval(val):
            if v['name'] not in crew_dict.keys():
                crew_dict[v['name']] = 1
            else:
                crew_dict[v['name']] += 1
    famous_crew = sorted(crew_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    for name, _ in famous_crew:
        train['famous-crew-' + name] = train['cast'].map(lambda x: exist_name(x, name))
        test['famous-crew-' + name] = test['cast'].map(lambda x: exist_name(x, name))
    # cast crew end

    # genres begin
    genres_dict = {}
    for val in train['genres']:
        for v in eval(val):
            if v['name'] not in genres_dict.keys():
                genres_dict[v['name']] = 1
            else:
                genres_dict[v['name']] += 1
    train['genres-all'] = train['genres'].map(lambda x: ''.join([v['name'] for v in eval(x)]))
    test['genres-all'] = test['genres'].map(lambda x: ''.join([v['name'] for v in eval(x)]))
    top_genres = ['Comedy', 'Romance', 'Action', 'Drama', 'Thriller']
    for genres in top_genres:
        train['genres-' + genres] = train['genres'].map(lambda x: 1 if genres in x else 0)
        test['genres-' + genres] = test['genres'].map(lambda x: 1 if genres in x else 0)
    # genres end

    # language begin
    languages_dict = {}
    for val in train['spoken-languages']:
        for v in eval(val):
            if v['iso_639_1'] not in languages_dict.keys():
                languages_dict[v['iso_639_1']] = 1
            else:
                languages_dict[v['iso_639_1']] += 1
    train['languages-all'] = train['spoken-languages'].map(lambda x: ''.join([v['iso_639_1'] + ' ' for v in eval(x)]))
    test['languages-all'] = test['spoken-languages'].map(lambda x: ''.join([v['iso_639_1'] + ' ' for v in eval(x)]))
    top_languages = ['en', 'fr', 'it', 'de', 'es', 'ru', 'zh', 'ja']
    for language in top_languages:
        train['languages-' + language] = train['languages-all'].map(lambda x: 1 if language in x else 0)
        test['languages-' + language] = test['languages-all'].map(lambda x: 1 if language in x else 0)
    # language end
    train.drop(['genres', 'original-language', 'production-companies', 'production-countries', 'spoken-languages'],
               axis=1, inplace=True)
    test.drop(['genres', 'original-language', 'production-companies', 'production-countries', 'spoken-languages'],
              axis=1, inplace=True)

    # keywords begin
    keywords_dict = {}
    for val in train['Keywords']:
        for v in eval(val):
            if v['name'] not in keywords_dict.keys():
                keywords_dict[v['name']] = 1
            else:
                keywords_dict[v['name']] += 1
    top_keywords = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)[:12]

    for keyword, _ in top_keywords:
        train['keywords-' + keyword] = train['Keywords'].map(lambda x: exist_name(x, keyword))
        test['keywords-' + keyword] = test['Keywords'].map(lambda x: exist_name(x, keyword))
    # keywords end
    train_texts = train[['original-title', 'overview', 'tagline', 'title', 'Keywords']]
    test_texts = test[['original-title', 'overview', 'tagline', 'title', 'Keywords']]
    for col in ['original-title', 'overview', 'tagline', 'title', 'Keywords']:
        train['words-' + col] = train[col].map(lambda x: len(str(x)))
        test['words-' + col] = test[col].map(lambda x: len(str(x)))
    train.drop(['original-title', 'overview', 'tagline', 'title', 'Keywords'], axis=1, inplace=True)
    test.drop(['original-title', 'overview', 'tagline', 'title', 'Keywords'], axis=1, inplace=True)


def lightgbm_train(min_data, epoch=1000):
    global train, target_name
    params_lgb = {
        'task': 'train',
        'objective': 'regression',
        'learning_rate': 0.01,
        'boosting': 'gbdt',
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'metric': 'rmse',
        'random_state': 2077
    }
    X_train, X_valid, y_train, y_valid = train_test_split(train.drop(target_name, axis=1), train[target_name].values, test_size=0.1, random_state=2077)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    regressor = lgb.train(params=params_lgb, train_set=lgb_train, valid_sets=lgb_valid, verbose_eval=50, valid_names='eval', num_boost_round=epoch)
    y_pred = regressor.predict(X_valid, num_iteration=regressor.best_iteration)
    print(y_pred)
    return regressor


def predict(regressor):
    global test
    pred = regressor.predict(test, num_iteration=regressor.best_iteration)
    pred = pred.astype(int)
    df = pd.DataFrame(pred)
    df.to_csv('prediction.csv', index=False)


def main():
    global train, test, target_name
    nni_params = nni.get_next_parameter()
    if 'sample_feature' in nni_params.keys():
        sample_col = nni_params['sample_feature']
    else:
        sample_col = []
    train = name2feature(train, sample_col, target_name)
    regressor = lightgbm_train(min_data=200)
    predict(regressor)


if __name__ == '__main__':
    fillna()
    train.to_csv('tmdb-box-office-prediction-train.csv', index=False)
    test.to_csv('tmdb-box-office-prediction-test.csv', index=False)
    main()
