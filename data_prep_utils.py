import pandas as pd
import numpy as np


def getClosestDate(team_attributes, match_date, team_id):
    # selecting team_attrivutes based on team id
    selection = team_attributes[team_attributes.team_api_id == team_id]

    # finding the closest date from selection
    closest_dt = None
    closest_days = 100000
    for date in selection.date:
        diff = abs((match_date-date).days)
        if diff < closest_days:
            closest_days = diff
            closest_dt = date
    return closest_dt


def merge_match_and_team(match_df, team_df):
    match_df['date'] =  pd.to_datetime(match_df['date'], format='%Y-%m-%d')
    team_df['date'] =  pd.to_datetime(team_df['date'], format='%Y-%m-%d')
    
    # creating array for each row that contains the closest_date
    closest_home_dates = []
    closest_away_dates = []
    for index, row in match_df.iterrows():
        match_date = row['date']
        home_team = row['home_team_api_id']
        away_team = row['away_team_api_id']
        
        # finding closest date for home team
        closest_home_dates.append(getClosestDate(team_df, match_date, home_team))
        # finding closest date for away team
        closest_away_dates.append(getClosestDate(team_df, match_date, away_team))
    
    # adding corresponding columns to the table
    match_df['closest_home_date'] = np.array(closest_home_dates)
    match_df['closest_away_date'] = np.array(closest_away_dates)
    
    # renameing team table columns so that they would have more logical column names in final table
    home_team_df = team_df.rename(columns=lambda x: "home_"+x)
    away_team_df = team_df.rename(columns=lambda x: "away_"+x)
    
    # currently just assuming that this line works as I wish
    # todo: check it later
    match_df = match_df.merge(home_team_df, left_on=['home_team_api_id', 'closest_home_date'],
                              right_on=['home_team_api_id', 'home_date'])
    # dropping unnecessary columns after merge
    match_df = match_df.drop(['home_team_api_id', 'closest_home_date', 'home_id', 
                              'home_team_fifa_api_id', 'home_date'], axis=1)
    
    # doing exactly the same thing with away team as I did previously with home team
    match_df = match_df.merge(away_team_df, left_on=['away_team_api_id', 'closest_away_date'],
                              right_on=['away_team_api_id', 'away_date'])
    # dropping unnecessary columns after merge
    match_df = match_df.drop(['away_team_api_id', 'closest_away_date', 'away_id', 
                              'away_team_fifa_api_id', 'away_date'], axis=1)

    return match_df
