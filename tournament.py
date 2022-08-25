from game import Game
import numpy as np
import copy 

class Tournament(Game):
    
    
    def __init__(self, games_per_group = 7, win_thresh = 4, random_state = None):

        self.games_per_group  = games_per_group
        self.win_thresh = win_thresh
        self.team_list = None
        self.rounds = {} # keep track the number of times a team wins at each round 
        super().__init__(random_state)

        
    '''
        This method simulates the entire playoffs by pairing the teams and future winners

    '''
    def simulate(self, DATA, generator_distros, pipeline, group_list, n_simulation, probs = True):
                     
        # update the list of teams
        self.rounds = {}
        self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        
        for i in range(n_simulation):
            cham = self.one_simulation(DATA, generator_distros, pipeline, group_list)
        if probs:
            self.rounds_probs =  self._compute_probs()
            

    '''
        This method simulates the entire playoff once, we may want to simulate it multiple times
    '''
    def one_simulation(self, DATA, generator_distros, pipeline, group_list, verbose = False, probs = False):
                
        # update the list of teams if haven't done so
        if self.team_list == None: 
            self.team_list = [i[0] for i in group_list] + [i[1] for i in group_list]
        round_number, done = 0, 0

        while not done: 
            all_group_winners, group_list = self.play_round(DATA, generator_distros, pipeline, group_list)
            # retrive round stats
            try:
                updated_round_stats = self.rounds[round_number]
            except KeyError:
                updated_round_stats = {}
                for team in self.team_list:
                    updated_round_stats[team] = 0
            # if a team wins, record + 1 
            for winner in all_group_winners:
                try: 
                    updated_round_stats[winner] += 1
                except KeyError:
                    pass     
            self.rounds[round_number] = updated_round_stats
            if verbose:
                print('{} round played'.format(round_number))
            if probs:
                self.rounds_probs = self._compute_probs()
            if type(group_list) != list: # if it becomes the final
                done = 1
            round_number += 1
            
        return group_list


    '''
        This method plays one round of the matched playoffs
    '''
    def play_round(self, DATA, generator_distros, pipeline, group_list):
        
        
        all_group_winners = [] 
        # play each group and get the group winner
        for group in group_list:
            winner = self.play_n_games(DATA, generator_distros, pipeline, group[0], group[1])
            all_group_winners.append(winner)
        
        if len(all_group_winners) > 1:
            new_group_list = []         
            for index in range(0, len(all_group_winners), 2):
                # first winner, second winner
                new_group = [all_group_winners[index], all_group_winners[index + 1]]
                new_group_list.append(new_group)
                
            return all_group_winners, new_group_list
        else:  
            return all_group_winners, winner
        
    '''
        This method plays all games in a given round, that is 7 games.
    '''
    def play_n_games(self, DATA, generator_distros, pipeline, team1, team2):
        
        
        result = Game().predict(team1=team1, team2=team2, DATA=DATA, generator_distros=generator_distros, pipeline=pipeline, num_games=self.games_per_group)
        if sum(result[:4]) == self.win_thresh or sum(result) >= self.win_thresh:
            winner = team1 # home team wins
        else:
            winner = team2 # visitor team wins
            
        return winner
    
    
    def _compute_probs(self):
                
        rounds_probs = copy.deepcopy(self.rounds)
        for round_number, round_stats in rounds_probs.items():
            m = np.sum(list(round_stats.values()))
            for k, v in rounds_probs[round_number].items():
                rounds_probs[round_number][k] = v / m
                
        return rounds_probs

    def get_round_probs(self):
        return self.rounds_probs