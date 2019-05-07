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
    
    # merging match data with team data
    match_df = match_df.merge(home_team_df, left_on=['home_team_api_id', 'closest_home_date'],
                              right_on=['home_team_api_id', 'home_date'])
    match_df = match_df.merge(away_team_df, left_on=['away_team_api_id', 'closest_away_date'],
                              right_on=['away_team_api_id', 'away_date'])
    
    # adding win and draw percentages from the last matches
    for match_value in ["3","5","10","20"]:
        # home team win
        match_df['home_win_last_' + match_value + '_matches'] = match_df.apply(lambda x:
                get_win_percent_in_last_n_matches(x['home_team_api_id'], x['home_date'], match_value, match_df), axis=1)
        # home draw
        match_df['home_draw_last_' + match_value + '_matches'] = match_df.apply(lambda x:
                get_draw_percent_in_last_n_matches(x['home_team_api_id'], x['home_date'], match_value, match_df), axis=1)

        # away win
        match_df['away_win_last_' + match_value + '_matches'] = match_df.apply(lambda x:
                get_win_percent_in_last_n_matches(x['away_team_api_id'], x['home_date'], match_value, match_df), axis=1)       
        # away draw
        match_df['away_draw_last_' + match_value + '_matches'] = match_df.apply(lambda x:
                get_draw_percent_in_last_n_matches(x['away_team_api_id'], x['home_date'], match_value, match_df), axis=1)
                
    # dropping unnecessary columns after merge
    match_df = match_df.drop(['home_team_api_id', 'closest_home_date', 'home_id', 
                              'home_team_fifa_api_id', 'home_date'], axis=1)
    match_df = match_df.drop(['away_team_api_id', 'closest_away_date', 'away_id', 
                              'away_team_fifa_api_id', 'away_date'], axis=1)

    return match_df


def get_win_percent_in_last_n_matches(team_id, match_date, n, match_t):
    n = int(n)
    # getting all the suitable matches that should be taken into consideration
    match_t['date'] =  pd.to_datetime(match_t['date'], format='%Y-%m-%d')
    match_t = match_t[((match_t.home_team_api_id == team_id) | (match_t.away_team_api_id == team_id)) & 
                      (match_t.date < match_date)]
    
    if match_t.shape[0] == 0:
        return 0.4
    
    # selectiong most recent n matches
    match_t = match_t.sort_values(by='date', ascending=False)
    match_t = match_t.head(n)
    
    # getting won matches from recent matches
    won_as_home_team = match_t[(match_t.home_team_goal > match_t.away_team_goal) & 
                               (match_t.home_team_api_id == team_id)]
    won_as_away_team = match_t[(match_t.away_team_goal > match_t.home_team_goal) & 
                               (match_t.away_team_api_id == team_id)]
    
    # return the win percent
    return (won_as_home_team.shape[0] + won_as_away_team.shape[0]) / match_t.shape[0]


def get_draw_percent_in_last_n_matches(team_id, match_date, n, match_t):
    n = int(n)
    # getting all the suitable matches that should be taken into consideration
    match_t['date'] =  pd.to_datetime(match_t['date'], format='%Y-%m-%d')
    match_t = match_t[((match_t.home_team_api_id == team_id) | (match_t.away_team_api_id == team_id)) & 
                      (match_t.date < match_date)]
    
    if match_t.shape[0] == 0:
        return 0.2
    
    # selectiong most recent n matches
    match_t = match_t.sort_values(by='date', ascending=False)
    match_t = match_t.head(n)
    
    # getting matches from recent matches taht ended with draw
    draw_t = match_t[match_t.home_team_goal == match_t.away_team_goal]
    
    # return the draw percent
    return draw_t.shape[0] / match_t.shape[0]

