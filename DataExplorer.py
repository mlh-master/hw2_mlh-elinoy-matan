import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def bin_to_num(df):
    df_bin = pd.DataFrame()
    for col in df:
        if col == 'Family History' or col == 'Age':
            df_bin.loc[:, col] = df.loc[:, col]
        elif col == 'Gender':
            df_bin.loc[:, col] = feat_to_num(df.loc[:, col], 'Male')
        elif col == 'Diagnosis':
            df_bin.loc[:, col] = feat_to_num(df.loc[:, col], 'Positive')
        else:
            df_bin.loc[:, col] = feat_to_num(df.loc[:, col], 'Yes')
    return df_bin

def feat_to_num(feat, label):
    """Turn binary information into a numeric
    :param: data (pd.dataframe)
    """
    feat = feat.eq(label).mul(1)
    return feat


def nan_explorer(data):
    nan_stat = {}
    for col in data:
        nan_stat[col] = data.loc[:, col].isna().sum()
    return nan_stat


def data_explorer(df):
    label = 'Diagnosis'
    Diagnosis = df.loc[:, label]
    legend = ['Positive', 'Negative']

    for col in df:
        if col != label:
            x1 = df.loc[Diagnosis == 'Positive', col]
            x2 = df.loc[Diagnosis == 'Negative', col]
            plt.hist([x1, x2], color=['blue', 'orange'], bins=4)

            plt.xlabel(col, fontsize=30)
            plt.ylabel('count', fontsize=30)
            plt.legend(legend)
            plt.show()

def data_gender_explorer(df):

    legend = ['female','male']

    x1 = df['Age'].loc[(df['Diagnosis'] == 'Positive') & (df['Gender'] == 'Female')]
    x2 = df['Age'].loc[(df['Diagnosis'] == 'Positive') & (df['Gender'] == 'Male')]
    plt.hist([x1,x2], color=['blue','orange'], bins =10)

    plt.xlabel('Age', fontsize=30)
    plt.ylabel('count', fontsize=30)
    plt.title ('Positive diagnosis')
    plt.legend(legend)
    plt.show()

def test_train_comparison(X_train, X_test):
    data_dict = {}

    for col in X_train:
        if col != 'Age':
            temp_dict = {'train %': round(100 * X_train.loc[:, col].sum() / len(X_train.loc[:, col])),
                         'test %': round(100 * X_test.loc[:, col].sum() / len(X_test.loc[:, col]))}
            data_dict[col] = temp_dict

    return pd.DataFrame.from_dict(data_dict, orient='index')


def encode_and_bind(df):
    for col in df:
        if col != 'Age':
            dummies = pd.get_dummies(df[[col]].astype(str))
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
    return df

def age_scale(X_train, X_test):
    scaler = StandardScaler()
    for col in X_train:
        if col == 'Age':
            train_Age_scale = scaler.fit_transform(X_train[[col]])
            X_train = X_train.assign(Age=train_Age_scale)
            test_Age_scale = scaler.transform(X_test[[col]])
            X_test = X_test.assign(Age=test_Age_scale)
    return X_train, X_test

