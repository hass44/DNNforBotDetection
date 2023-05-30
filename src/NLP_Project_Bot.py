
import pandas as pd
import numpy as np
import math
from keras.layers import Embedding, Dense, LSTM, Dense, Input, concatenate
from keras.models import Model
from keras.utils import plot_model
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.utils import  pad_sequences
import pandas as pd

bot_accounts = pd.concat([pd.read_csv('D:/DNNforBotDetection/DNNforBotDetection/input/cresci-2017.csv/social_spambots_1/users.csv'),
                         pd.read_csv('D:/DNNforBotDetection/DNNforBotDetection/input/cresci-2017.csv/social_spambots_2/users.csv'),
                         pd.read_csv('D:/DNNforBotDetection/DNNforBotDetection/input/cresci-2017.csv/social_spambots_3/users.csv')]).reset_index(drop=True)
clean_accounts = pd.read_csv('D:/DNNforBotDetection/DNNforBotDetection/input/cresci-2017.csv/genuine_accounts/users.csv')

requiredColumns = ['screen_name', 'created_at', 'updated', 'location', 'statuses_count', 'friends_count',
                   'followers_count', 'favourites_count', 'default_profile_image', 'profile_use_background_image',
                   'protected', 'default_profile']
bot_accounts = bot_accounts[requiredColumns]
clean_accounts = clean_accounts[requiredColumns]

def clean_df(df):
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None).dt.tz_localize('UTC')  # Convert to UTC timezone
    df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None).dt.tz_localize('UTC')  # Convert to UTC timezone
    df['age'] = (df['updated'] - df['created_at']).dt.days  # Use dt.days to get the number of days
    df['has_location'] = df['location'].notnull().astype(int)  # Convert boolean to integer
    df['has_avatar'] = df['default_profile_image'].notnull().astype(int)
    df['has_background'] = df['profile_use_background_image'].notnull().astype(int)
    df['is_protected'] = df['protected'].notnull().astype(int)
    df['profile_modified'] = df['default_profile'].notnull().astype(int)
    df = df.rename(columns={"screen_name": "username", "statuses_count": "total_tweets", "friends_count": "total_following",
                            "followers_count": "total_followers", "favourites_count": "total_likes"})
    return df[['username', 'age', 'has_location', 'total_tweets', 'total_following', 'total_followers',
               'total_likes', 'has_avatar', 'has_background', 'is_protected', 'profile_modified']]

bot_accounts = clean_df(bot_accounts)
clean_accounts = clean_df(clean_accounts)


bot_accounts['BotOrNot'] = 1
clean_accounts['BotOrNot'] = 0

combined_df = pd.concat([bot_accounts, clean_accounts])

new_df = combined_df.sample(frac=1).reset_index(drop=True)





training_df = new_df.drop('username', axis=1)[:int(combined_df.shape[0] * 0.8)]
test_df = new_df.drop('username', axis=1)[int(combined_df.shape[0] * 0.8):]

columns_to_standardize = ['age', 'total_tweets', 'total_following', 'total_followers', 'total_likes']

training_df_mean = training_df[columns_to_standardize].mean()
training_df_std = training_df[columns_to_standardize].std()

training_df[columns_to_standardize] = (training_df[columns_to_standardize] - training_df_mean)/training_df_std
test_df[columns_to_standardize] = (test_df[columns_to_standardize] - training_df_mean)/training_df_std





X_train = training_df.drop(['BotOrNot', 'is_protected'], axis=1).values
y_train = training_df['BotOrNot'].values.reshape(-1,1)

X_test = test_df.drop(['BotOrNot', 'is_protected'], axis=1).values
y_test = test_df['BotOrNot'].values.reshape(-1,1)



from imblearn.over_sampling import SMOTE

s = SMOTE()
smote_X, smote_y = s.fit_resample(X_train, y_train.reshape(-1))





def clean(df):
    
    df = df.drop('username', axis=1)
    
    df[columns_to_standardize] = (df[columns_to_standardize] - training_df_mean)/training_df_std
    df = df.drop('is_protected', axis=1)
    return df



