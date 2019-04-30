# takes in dataframe with "home_team_goal" and "away_team_goal" columns
# converts mentioned columns into -1 (away team won), 0 (draw) and 1 (home_team_won)
def make_goals_into_win_label(df):
    #data = data.assign(label= lambda x: assign_value(x))
    df['label'] = -1
    df.loc[df.home_team_goal > df.away_team_goal, 'label'] = 1
    df.loc[df.home_team_goal == df.away_team_goal, 'label'] = 0
    df.drop(['home_team_goal', 'away_team_goal'], axis=1, inplace=True)
    return df
