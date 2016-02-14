#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import util_logging


logger = util_logging.get_logger(__name__)

class Dataset:
    def __init__(self, name, input_dir='./input', model_dir='./models'):
        self.name = name
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.target = 'country_destination'
        self.features = []
        self.train_users = pd.DataFrame()
        self.test_users = pd.DataFrame()
        self.sessions = pd.DataFrame()
        self.combined = pd.DataFrame()
        self.le_target = LabelEncoder()

    def save_pkl(self):
        logger.info('Saving...')
        with open('{}/{}.pkl'.format(self.model_dir, self.name), 'wb') as f:
            pickle.dump(self, f)
        logger.info('Saved')

    def load_pkl(self):
        logger.info('Loading...')
        with open('{}/{}.pkl'.format(self.model_dir, self.name), 'rb') as f:
            dataset = pickle.load(f)
        logger.info('Done')
        return dataset

    def generate(self):
        logger.info('Generate Dataset...')
        self.__load_input()
        self.__prepare()
        self.__preproc()
        self.__feature_engineering()
        self.__preproc_sessions()
        self.__feature_selection()
        self.__train_test_split()
        logger.info('Train Shape: {} Test Rows: {}'.format(self.train.shape[0],
                                                           self.test.shape[0]))
        logger.info('Number of Features: {}'.format(len(self.features)))
        logger.info('Finished')

    def __load_input(self):
        self.train_users = pd.read_csv('{}/{}'.format(self.input_dir, 'train_users_2.csv'), parse_dates=[1, 2, 3])
        self.test_users = pd.read_csv('{}/{}'.format(self.input_dir, 'test_users.csv'), parse_dates=[1, 2, 3])
        self.sessions = pd.read_csv('{}/{}'.format(self.input_dir, 'sessions.csv'))

    def __prepare(self):
        self.le_target.fit(self.train_users['country_destination'].unique())
        self.train_users['country_destination'] = self.le_target.transform(
            self.train_users['country_destination'].values)
        self.test_users['country_destination'] = -1
        self.combined = pd.concat([self.train_users, self.test_users], ignore_index=True)

    def __preproc(self):
        # age
        self.combined['age'].fillna(-1, inplace=True)
        self.combined['age'] = np.where(self.combined['age'] < 14, -1, self.combined['age'])
        self.combined['age'] = np.where(self.combined['age'] > 100, -1, self.combined['age'])
        # first_affiliate_tracked
        self.combined['first_affiliate_tracked'].fillna(-1, inplace=True)

    def __feature_engineering(self):
        # number of nas
        self.combined['n_nas_profile'] = np.sum([(self.combined['age'] == -1),
                                                 (self.combined['gender'] == '-unknown-'),
                                                 (self.combined['language'] == '-unknown-')], axis=0)
        self.combined['n_nas'] = np.sum([(self.combined['age'] == -1),
                                         (self.combined['gender'] == '-unknown-'),
                                         (self.combined['language'] == '-unknown-'),
                                         (self.combined['first_affiliate_tracked'] == -1),
                                         (self.combined['first_affiliate_tracked'] == 'untracked'),
                                         (self.combined['first_browser'] == '-unknown-'),], axis=0)

        # date_account_created
        self.combined['dac_year'] = self.combined['date_account_created'].dt.year
        self.combined['dac_quarter'] = self.combined['date_account_created'].dt.quarter
        self.combined['dac_month'] = self.combined['date_account_created'].dt.month
        self.combined['dac_weekofyear'] = self.combined['date_account_created'].dt.weekofyear
        self.combined['dac_dayofyear'] = self.combined['date_account_created'].dt.dayofyear
        self.combined['dac_dayofweek'] = self.combined['date_account_created'].dt.dayofweek
        self.combined['dac_day'] = self.combined['date_account_created'].dt.day

        # timestamp_first_active
        self.combined['tfa_year'] = self.combined['timestamp_first_active'].dt.year
        self.combined['tfa_quarter'] = self.combined['timestamp_first_active'].dt.quarter
        self.combined['tfa_month'] = self.combined['timestamp_first_active'].dt.month
        self.combined['tfa_weekofyear'] = self.combined['timestamp_first_active'].dt.weekofyear
        self.combined['tfa_dayofyear'] = self.combined['timestamp_first_active'].dt.dayofyear
        self.combined['tfa_dayofweek'] = self.combined['timestamp_first_active'].dt.dayofweek
        self.combined['tfa_day'] = self.combined['timestamp_first_active'].dt.day
        self.combined['tfa_hour'] = self.combined['timestamp_first_active'].dt.hour

        # datetime delta
        BASE_DATE = datetime.date(2009, 1, 1)
        self.combined['days_dac_bd'] = (self.combined['date_account_created'] - BASE_DATE).dt.days
        self.combined['days_tfa_bd'] = (self.combined['timestamp_first_active'] - BASE_DATE).dt.days

        self.combined['days_dac_tfa'] = (self.combined['date_account_created'] - \
            self.combined['timestamp_first_active'] + datetime.timedelta(days=1)).dt.days

    def __preproc_sessions(self):
        action_columns = ['action', 'action_type', 'action_detail']
        self.sessions[action_columns] = self.sessions[action_columns].fillna('MISSING')

        self.sessions['counts'] = 1

        se25 = self.sessions['secs_elapsed'].quantile(0.25)
        self.sessions['secs_elapsed25'] = np.where(self.sessions['secs_elapsed'] > se25,
                                                   se25, self.sessions['secs_elapsed'])
        se50 = self.sessions['secs_elapsed'].quantile(0.50)
        self.sessions['secs_elapsed50'] = np.where(self.sessions['secs_elapsed'] > se50,
                                                   se50, self.sessions['secs_elapsed'])

        columns = ['user_id', 'action', 'action_type', 'action_detail', 'device_type']
        sessions_count = self.sessions.groupby(columns)[['counts']].count()
        sessions_count.reset_index(inplace=True)

        ohe_columns = ['action', 'action_type', 'action_detail', 'device_type']
        added_columns = []
        for col in ohe_columns:
            dummy = pd.get_dummies(sessions_count[col], prefix=col)
            added_columns += dummy.columns.tolist()
            sessions_count.drop([col], axis=1, inplace=True)
            sessions_count = pd.concat([sessions_count, dummy], axis=1)

        sessions_count[added_columns] = sessions_count[added_columns].multiply(sessions_count['counts'], axis='index')

        sessions_user = sessions_count.groupby('user_id').sum()

        self.sessions.drop(['counts'], axis=1, inplace=True)

        # secs_elapsed
        df = self.sessions.groupby('user_id').mean().add_suffix('_mean')
        sessions_user = pd.merge(sessions_user, df, how='left', left_index=True, right_index=True)
        df = self.sessions.groupby('user_id').sum().add_suffix('_sum')
        sessions_user = pd.merge(sessions_user, df, how='left', left_index=True, right_index=True)
        df = self.sessions.groupby('user_id').std().add_suffix('_std')
        sessions_user = pd.merge(sessions_user, df, how='left', left_index=True, right_index=True)
        df = self.sessions.groupby('user_id').median().add_suffix('_median')
        sessions_user = pd.merge(sessions_user, df, how='left', left_index=True, right_index=True)

        self.combined = pd.merge(self.combined, sessions_user, how='left', left_on='id', right_index=True)
        columns = sessions_user.columns
        self.combined[columns] = self.combined[columns].fillna(-1)

    def __feature_selection(self):
        unused_columns = ['id', 'date_account_created', 'timestamp_first_active',
                          'date_first_booking', 'country_destination']
        for col in self.combined.columns:
            if col in unused_columns:
                continue
            self.features.append(col)
            if self.combined[col].dtype == 'object':
                le = LabelEncoder()
                le.fit(self.combined[col].values)
                self.combined[col] = le.transform(self.combined[col].values)

    def __train_test_split(self):
        self.train = self.combined[self.combined['country_destination'] != -1]
        self.test = self.combined[self.combined['country_destination'] == -1]


if __name__ == '__main__':
    dataset = Dataset(name='b88369b')
    dataset.generate()
    dataset.save_pkl()
