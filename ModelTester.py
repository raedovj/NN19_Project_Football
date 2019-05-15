import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical


class ModelTester():

    def __init__(self, model, x_train, y_train, x_test, y_test, bet_train, bet_test):
        self.bet_train = bet_train
        self.bet_test = bet_test
        
        self.y_train = y_train
        self.y_test = y_test
        
        self.train_predictions_3x1 = model.predict(x_train)
        self.test_predictions_3x1 = model.predict(x_test)
        self.train_predictions = np.argmax(self.train_predictions_3x1 , axis=1) - 1
        self.test_predictions = np.argmax(self.test_predictions_3x1 , axis=1) - 1
        
        
        
    # ===========================================================================================================
    # most important functions here
    
    def get_train_confusion_matrix(self):
        return pd.DataFrame(confusion_matrix(self.y_train, self.train_predictions, labels=[-1, 0, 1]), 
             index=['true home', 'true draw', 'true away'], columns=['pred home', 'pred draw', 'pred away'])
    
    def get_test_confusion_matrix(self):
        return pd.DataFrame(confusion_matrix(self.y_test, self.test_predictions, labels=[-1, 0, 1]), 
             index=['true home', 'true draw', 'true away'], columns=['pred home', 'pred draw', 'pred away'])
    
    def get_train_accuracy(self):
        return (self.train_predictions == self.y_train.values).sum() / self.y_train.values.shape[0]
    
    def get_test_accuracy(self):
        return (self.test_predictions == self.y_test.values).sum() / self.y_test.values.shape[0]
     
    def always_bet_predicted_winner_profit(self, data="test"):
        if data == "test":
            self.always_bet_predicted_winner_profit_helper(self.test_predictions, self.y_test, self.bet_test)
        else:
            self.always_bet_predicted_winner_profit_helper(self.train_predictions, self.y_train, self.bet_train)
            
    def bet_predicted_winner_with_threshold_profit(self, data="test", threshold=1):
        if data == "test":
            self.bet_predicted_winner_with_threshold_profit_helper(self.test_predictions_3x1, self.test_predictions, 
                                                                   self.y_test, self.bet_test, threshold)
        else:
            self.bet_predicted_winner_with_threshold_profit_helper(self.train_predictions_3x1, self.train_predictions,
                                                                   self.y_train, self.bet_train, threshold)
            
    def predict_on_highest_return(self, data="test", threshold=2.5):
        if data == "test":
            self.predict_on_highest_return_helper(self.test_predictions_3x1, self.y_test, self.bet_test, threshold)
        else:
            self.predict_on_highest_returnt_helper(self.train_predictions_3x1, self.y_train, self.bet_train, threshold)

            
            
        
    # ===========================================================================================================
    # just statistics functions and sanity checks, which might be useful
    
    
    # always betting on one thing
    # by default doing it on test set.
    def predict_home_wins_only_profit(self, data="test"):
        self.predict_always_on_one_thing_benefit_helper(data, 0)
        
    def predict_draw_only_profit(self, data="test"):
        self.predict_always_on_one_thing_benefit_helper(data, 1)
        
    def predict_away_wins_only_profit(self, data="test"):
        self.predict_always_on_one_thing_benefit_helper(data, 2)
        
    def predict_bet_home_away_profit(self, data="test"):
        self.predict_always_on_one_thing_benefit_helper(data, [0,2])
        
    def predict_bet_on_all_profit(self, data="test"):
        self.predict_always_on_one_thing_benefit_helper(data, [0,1,2])

        
        
    # getting different win ratios
    def get_test_home_team_win_rate(self):
        return (-1 == self.y_test.values).sum() / self.y_test.values.shape[0]
    
    def get_train_home_team_win_rate(self):
        return (-1 == self.y_train.values).sum() / self.y_train.values.shape[0]
    
    def get_test_draw_rate(self):
        return (0 == self.y_test.values).sum() / self.y_test.values.shape[0]
    
    def get_train_draw_rate(self):
        return (0 == self.y_train.values).sum() / self.y_train.values.shape[0]
    
    def get_test_away_team_win_rate(self):
        return (1 == self.y_test.values).sum() / self.y_test.values.shape[0]
    
    def get_train_away_team_win_rate(self):
        return (1 == self.y_train.values).sum() / self.y_train.values.shape[0]

    
    
    # ===========================================================================================================
    # helpful functions for main functions
    
    def agencies(self):
        return ['B365', 'BW', 'IW', 'LB', 'WH', 'SJ', 'VC', 'GB', 'BS']
    
    def predict_always_on_one_thing_benefit_helper(self, data, index):
        if data == "test":
            self.predict_always_on_one_thing_benefit(self.y_test, self.bet_test, index)
        else:
            self.predict_always_on_one_thing_benefit(self.y_train, self.bet_train, index)
    
    def predict_always_on_one_thing_benefit(self, labels, betting_odds, predictable_value):
        predictable_indices = np.zeros((labels.shape[0], 3))
        predictable_indices[:, predictable_value] = 1

        for agency in self.agencies():
            odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
            # r holds betting results. 0 indicates loss, value otherwise shows the win amount
            r = odds * predictable_indices * to_categorical(labels+1)
            # Take max value of win, draw, other win. 
            r = r.values.max(axis=1)
            # Let's say we bet 1€. then our profit(or loss) would be = earnings€ - 1€ per bet
            r -= len(predictable_value)
       
            print("Agency %s, \twin amount: %.2f" % (agency, r.sum()))
            
    def always_bet_predicted_winner_profit_helper(self, predictions, labels, betting_odds):      
        predictions_categorical = to_categorical(predictions + 1)
        for agency in self.agencies():
            odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
            # r holds betting results. 0 indicates loss, value otherwise shows the win amount
            r = odds * predictions_categorical * to_categorical(labels+1)
            # Take max value of win, draw, other win. 
            r = r.values.max(axis=1)
            # Let's say we bet 1€. then our profit(or loss) would be = earnings€ - 1€
            r -= 1
            print("Agency %s, \twin amount: %.2f" % (agency, r.sum()))
            
    def bet_predicted_winner_with_threshold_profit_helper(self, predictions_3x1, predictions, labels, 
                                                          betting_odds, threshold):
        predictions_categorical = to_categorical(predictions + 1)
        for agency in self.agencies():
            odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
            # r holds betting results. 0 indicates loss, value otherwise shows the win amount
            bet = odds * predictions_categorical * predictions_3x1
            bet = bet > threshold
            r = odds * predictions_categorical * to_categorical(labels+1)
            r -= 1
            # Set win/lose amount to 0 on matched it didn't bet
            r[np.invert(bet)] = 0
            # Take max value of win, draw, other win. 
            r = r.values.max(axis=1)
            
            skip_percentage = (r==0).sum() / r.shape[0] * 100   
            print("Agency %s, \twin amount: %.2f. Didn't bet on %.2f%% of matches" % (agency, r.sum(), skip_percentage))
    
    # Set threshold to 0, to bet on all matches
    def predict_on_highest_return_helper(self, predictions_3x1, labels, betting_odds, threshold):
        for agency in self.agencies():
            odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
            # Expected earning value. Basically expects that our NN predicts real match outcomes
            expected = (odds * predictions_3x1).values

            # Threshold matches, when we'd actually would make a bet. If expected yield is too low, it'll pass
            bet = np.max(expected > threshold, axis=1)

            # Take the highest yield of [home win, draw, other win]
            r = np.argmax(expected, axis=1) 

            # Calculate wins/losses according to real match results
            r = to_categorical(r) * to_categorical(labels+1)
            r -= 1 # subtract our input bet

            # Calculate earnings
            r = r.max(axis=1) # Take max value of win, draw, other win. 
            r[np.invert(bet)] = 0 # Set win/lose amount to 0 on matched it didn't bet

            skip_percentage = (bet==0).sum() / bet.shape[0] * 100   
            print("Agency %s, \twin amount: %.2f. Didn't bet on %.2f%% of matches" % (agency, r.sum(), skip_percentage))     
                                  