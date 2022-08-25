import numpy as np

class Game:
    

    def __init__ (self, random_state = None):
        
        self.random_state = random_state # keep this to None for making simulations 
    
    '''
        This method predicts the winner of a game based on features sampled from their respective distributions.
    '''
    def predict(self, team1, team2, DATA, generator_distros, pipeline, num_games = 1):
        
        assert num_games >= 1
        # output numpy array
        team_1_features = DATA[team1]
        team_2_features = DATA[team2]

        features = []

        for team1_feature_params in team_1_features:
            sample_1 = self.sampling(team1_feature_params, generator_distros=generator_distros, size = num_games) # gives a list if num_games> 1
            features.append(sample_1) 
            
        for team2_feature_params in team_2_features:
            sample_2 = self.sampling(team2_feature_params, generator_distros=generator_distros, size = num_games) # gives a list if num_games> 1
            features.append(sample_2)
            
        features = np.array(features).T 
        win_loss = pipeline.predict(features)
        
        return list(win_loss) # a list of win/loss from num_games
    
    
    def sampling(self, dic, generator_distros, size = 1, random_state = None):
        
                        
        dis_name = list(dic.keys())[0] # get the type
        params = list(dic.values())[0] # get the params

        first_val = list(params.values())[0]
        second_val = list(params.values())[1]
    
        # get sample
        sample = generator_distros[dis_name](first_val, second_val, size = size,  random_state =  random_state)
        return sample 



