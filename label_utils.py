import pandas as pd
import numpy as np

# takes in dataframe with "home_team_goal" and "away_team_goal" columns
# converts mentioned columns into -1 (away team won), 0 (draw) and 1 (home_team_won)
def goals_to_single_number_discrete(df):
    #data = data.assign(label= lambda x: assign_value(x))
    df['label'] = 1
    df.loc[df.home_team_goal > df.away_team_goal, 'label'] = -1
    df.loc[df.home_team_goal == df.away_team_goal, 'label'] = 0
    df.drop(['home_team_goal', 'away_team_goal'], axis=1, inplace=True)
    return df

# Combines "home_team_goal" and "away_team_goal" columns by subtracting these
def goals_to_single_number_continuous(df):
    #data = data.assign(label= lambda x: assign_value(x))
    df['label'] = df.home_team_goal - df.away_team_goal
    df.drop(['home_team_goal', 'away_team_goal'], axis=1, inplace=True)
    return df

# Converts betting agency [Home, Draw, Away] multipliers to one number. Assumes
# these values to be inverse of probabilities and calculates expected value based on it.
def betting_agency_to_single_number(df):
    agencies = ['B365', 'BW', 'IW', 'LB', 'WH', 'SJ', 'VC', 'GB', 'BS']
    for agency in agencies:
        home, away, draw = 1/df[agency+'H'], 1/df[agency+'A'], 1/df[agency+'D'] 
        Sum = home+away+draw
        # Convert home and away bets to probabilities
        home /= Sum 
        away /= Sum
        # Store expected match outcome: 1*p(home) + -1*p(away)
        df[agency] = home - away
        df.drop([agency+'H', agency+'A', agency+'D'], axis=1, inplace=True)
    return df

def normalize_numeric(df):
    # https://stackoverflow.com/questions/44639442/pandas-ignore-string-columns-while-doing-normalization
    df_num = df.select_dtypes(include=[np.number])
    df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min())
    df[df_norm.columns] = df_norm
    return df

def normalize_not_in_range(df, a, b, param = None):
    params = {}
    for column_name in list(df):
        if not df[column_name].between(a, b).all():
            if param is None: 
                mean, maximum, minimum = df[column_name].mean(), df[column_name].max(), df[column_name].min()
                params[column_name] = (mean, maximum, minimum)
            else:
                mean, maximum, minimum = param[column_name]
            df[column_name] = (df[column_name] - mean) / (maximum - minimum)
    return df, params

def strings_to_numeric(df):
    label_to_num_map = {
     'Balanced':0, 'Fast':1,'Slow':-1,
     'Little':-1,'Normal':0,'Lots':1,
     'Organised':1,'Free Form':-1,
     'High':1,'Medium':0,'Deep':-1,
     'Press':0,'Double':1,'Contain':-1,
     'Wide':1,'Normal':0,'Narrow':-1,
     'Cover':-1,'Offside Trap':1,
     'Mixed':0, 'Long':1, 'Short':-1,
     'Risky':1, 'Safe':-1}
    
    # Convert strings to numeric data
    df = df.applymap(lambda s: label_to_num_map.get(s) 
                if (isinstance(s, str) and s in label_to_num_map) 
                else s)    
    return df
    
def separate_betting_agencies(df):
    new_df = pd.DataFrame()
    agencies = ['B365', 'BW', 'IW', 'LB', 'WH', 'SJ', 'VC', 'GB', 'BS']
    for agency in agencies:
        new_df = pd.concat([new_df, df[agency+'H'], df[agency+'D'], df[agency+'A']], axis=1)
        df.drop([agency+'H', agency+'D', agency+'A'], axis=1, inplace=True)
    return df, new_df
    
def process_data(df):
    df.drop(['league', 'country'], axis=1, inplace=True)

    # Drop columns, where over half of data is NaN
    df.dropna(axis='columns', thresh = int(df.shape[0]*0.5), inplace=True)
    
    #df = betting_agency_to_single_number(df)
    df = strings_to_numeric(df)
    df = goals_to_single_number_discrete(df)
    
    df.fillna(df.mean(), inplace=True) # Replace NaN with column means
    
    labels = df.label
    #df = normalize_not_in_range(df, -1, 1)
    df.label = labels # keep labels unnormalised
    
    return df

def split_data(df):
    train = df.sample(frac=0.8, random_state=123) # 123 is seed
    test = df.drop(train.index)
    
    x_train = train.drop('label', axis=1)
    y_train = train['label']
    x_test = test.drop('label', axis=1)
    y_test = test['label']
    
    _, bet_train = separate_betting_agencies(x_train.copy())
    _, bet_test = separate_betting_agencies(x_test.copy())
    
    x_train, norm_params = normalize_not_in_range(x_train, -1, 1)
    x_test, _ = normalize_not_in_range(x_test, -1, 1, norm_params)
    
    return x_train, y_train, x_test, y_test, bet_train, bet_test

def split_data_split_bet_agency(df):
    train = df.sample(frac=0.8, random_state=123) # 123 is seed
    test = df.drop(train.index)
    
    x_train = train.drop('label', axis=1)
    y_train = train['label']
    x_test = test.drop('label', axis=1)
    y_test = test['label']
    
    x_train, bet_train = separate_betting_agencies(x_train)
    x_test, bet_test = separate_betting_agencies(x_test)
    
    x_train, norm_params = normalize_not_in_range(x_train, -1, 1)
    x_test, _ = normalize_not_in_range(x_test, -1, 1, norm_params)
    
    return x_train, y_train, x_test, y_test, bet_train, bet_test