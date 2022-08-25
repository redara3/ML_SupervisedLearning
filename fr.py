from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

class featureReduction(object):

    def __init__(self):
        self.df = None
        self.gamesdf2 = None
        self.tempdf = None
        self.homedf = None
        self.tempy = None
        self.yhome = None
        self.target = None

        self.lreg = LinearRegression()

        self.gamesdf = pd.read_csv ("./data/games.csv")
        self.teamsdf = pd.read_csv ("./data/teams.csv")


    def makeDataFrames(self):

        df = pd.DataFrame({})
        df = df.append(self.gamesdf.iloc[0])

        self.gamesdf2 = pd.DataFrame({})
        for i in range(len(self.gamesdf)):
            if 2020 == self.gamesdf['SEASON'][i] and "Final" == self.gamesdf['GAME_STATUS_TEXT'][i]:
                self.gamesdf2 = self.gamesdf2.append(self.gamesdf.iloc[i])

        gamesdf3 = self.gamesdf2.drop("GAME_ID", axis = 1)
        gamesdf3 = gamesdf3.drop("GAME_STATUS_TEXT", axis = 1)
        gamesdf3 = gamesdf3.drop("SEASON", axis = 1)
        gamesdf3 = gamesdf3.drop("GAME_DATE_EST", axis = 1)
        gamesdf3 = gamesdf3.drop("HOME_TEAM_ID", axis = 1)
        gamesdf3 = gamesdf3.drop("VISITOR_TEAM_ID", axis = 1)

        tempdf = gamesdf3.drop("TEAM_ID_home",  axis = 1)
        tempdf = tempdf.drop("TEAM_ID_away",  axis = 1)
        tempdf = tempdf.drop("PTS_home",  axis = 1)
        self.tempdf = tempdf.drop("PTS_away",  axis = 1)
        # print(tempdf.iloc[0])

        homedf = gamesdf3.drop("TEAM_ID_home",  axis = 1)
        homedf = homedf.drop("TEAM_ID_away",  axis = 1)
        homedf = homedf.drop("PTS_away",  axis = 1)
        homedf = homedf.drop("FG_PCT_away",  axis = 1)
        homedf = homedf.drop("FT_PCT_away",  axis = 1)
        homedf = homedf.drop("FG3_PCT_away",  axis = 1)
        homedf = homedf.drop("AST_away",  axis = 1)
        self.homedf = homedf.drop("REB_away",  axis = 1)

        return homedf


    def getCorrelationMatrix(self, pdf):
        corrleation_matrix = pdf.corr()
        f, ax = plt.subplots(figsize=(15,10))
        sns.heatmap(corrleation_matrix, vmax=.8, square=True)


    def train(self):
        self.tempy = self.tempdf["HOME_TEAM_WINS"]
        self.tempdf = self.tempdf.drop("HOME_TEAM_WINS",  axis = 1)

        sfs1 = sfs(self.lreg, k_features=6, forward=False, verbose=1, scoring='neg_mean_squared_error') 
        sfs1 = sfs1.fit(self.tempdf, self.tempy)

        self.printFeatureNames(sfs1)

    def train2(self):
        self.yhome = self.homedf["HOME_TEAM_WINS"]
        self.homedf = self.homedf.drop("HOME_TEAM_WINS",  axis = 1)

        sfs2 = sfs(self.lreg, k_features=3, forward=False, verbose=1, scoring='neg_mean_squared_error') 
        sfs2 = sfs2.fit(self.homedf, self.yhome)

        self.printFeatureNames(sfs2)


    def backward_elimination(self, data, target,significance_level = 0.05):
        features = data.columns.tolist()
        while(len(features)>0):
            features_with_constant = sm.add_constant(data[features])
            p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if(max_p_value >= significance_level):
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
            else:
                break 
        return features

    def printFeatureNames(self, features):
        feat_names = features.k_feature_names_

        print(feat_names)