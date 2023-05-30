from twitter_scraper_selenium import get_profile_details
import pandas as pd
import json
from datetime import datetime


# In[ ]:


def getUserData(string):
    twitter_username = string.split('/')[-1]
    filename = "twitter_api_data"
    get_profile_details(twitter_username=twitter_username, filename=filename)
    
    with open('twitter_api_data.json', 'r') as file:
        json_data = file.read()
    json_list = json.loads(json_data)
    df = pd.json_normalize(json_list)
    
    requiredColumns = ['screen_name', 'created_at', 'location', 'verified', 'statuses_count', 'friends_count',
                   'followers_count', 'favourites_count', 'default_profile_image', 'profile_use_background_image',
                   'protected', 'default_profile']

    df = df[requiredColumns]
    
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.loc[:,'updated'] = current_datetime
    
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None).dt.tz_localize('UTC')  # Convert to UTC timezone
    df['updated'] = pd.to_datetime(df['updated']).dt.tz_localize(None).dt.tz_localize('UTC')  # Convert to UTC timezone
    df['age'] = (df['updated'] - df['created_at']).dt.days  # Use dt.days to get the number of days
    df['has_location'] = df['location'].notnull().astype(int)  # Convert boolean to integer
    df['has_avatar'] = df['default_profile_image'].notnull().astype(int)
    df['has_background'] = df['profile_use_background_image'].notnull().astype(int)
    df['is_verified'] = df['verified'].notnull().astype(int)
    df['is_protected'] = df['protected'].notnull().astype(int)
    df['profile_modified'] = df['default_profile'].notnull().astype(int)
    df = df.rename(columns={"screen_name": "username", "statuses_count": "total_tweets", "friends_count": "total_following",
                            "followers_count": "total_followers", "favourites_count": "total_likes"})
    
    return df[['username', 'age', 'has_location', 'is_verified', 'total_tweets', 'total_following', 'total_followers',
               'total_likes', 'has_avatar', 'has_background', 'is_protected', 'profile_modified']]





