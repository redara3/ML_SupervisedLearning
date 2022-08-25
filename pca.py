# %%

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_dataframe(df):
    df2 = df[:-1] #removes the final row containing average for all teams combined
    df2 = df.drop('G', axis = 1) #removal of columns that i thought were unimportant, like wins, losses, and minutes played
    df2 = df.drop('Rk', axis = 1)
    df2 = df2.drop("Team", axis = 1)
    df2 = df2.drop("MP", axis = 1)
    df2 = df2.drop('index', axis = 1)
    df2 = df2.dropna()#dropped null values in the columns

    return df2

def fit(dataframe):
    pca2 = PCA(n_components=7) #did pca
    pca2.fit(dataframe)

    return pca2

def visualize_variance(pca):
    #code to plot scree plot
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.bar(PC_values, pca.explained_variance_ratio_, linewidth=2, color='red')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()


def compute_components(pca):
    #code to compute what percentage of each feature was used to compute this component
    firstcomp = []
    for i in range(len(abs(pca.components_)[0])):
        firstcomp.append((i, abs(pca.components_)[0][i]))
    firstcomp.sort(key = lambda x: x[1], reverse = True)
    print(firstcomp)
    print(' ')
    #list containing indices of the features in firstcomp
    firstcompindices = []
    for atup in firstcomp:
        firstcompindices.append(atup[0])        

    #repeat of above process for each of the reamining features. i copy pasted the code to make visualization
    #of inidividual components easier

    secondcomp = []
    for i in range(len(abs(pca.components_)[1])):
        secondcomp.append((i, abs(pca.components_)[1][i]))
    secondcomp.sort(key = lambda x: x[1], reverse = True)

    secondcompindices = []
    for atup in secondcomp:
        secondcompindices.append(atup[0])
    
    print(secondcomp)
    print(' ')

    thirdcomp = []
    for i in range(len(abs(pca.components_)[2])):
        thirdcomp.append((i, abs(pca.components_)[2][i]))
    thirdcomp.sort(key = lambda x: x[1], reverse = True)

    thirdcompindices = []
    for atup in thirdcomp:
        thirdcompindices.append(atup[0])

    print(thirdcomp)
    print(' ')
  
    return (firstcompindices[:10], secondcompindices[:10], thirdcompindices[:10])

#variance_covered_by_components = sum(pca.explained_variance_ratio_[:5])
#print("variance covered by the 5 components" + " = " + str(variance_covered_by_components))

def select_features(components):
    # Make a frequency chart and select the top 10 features
    component_matrix = [components[0][:10], components[1][:10], components[2][:10], components[3][:10]]
    unique, counts = np.unique(component_matrix, return_counts=True)
    sorted_features = sorted(np.asarray([unique, counts]).T.tolist(), key=lambda x:x[1], reverse=True)[:15]
    selected_features = sorted([row[0] for row in sorted_features])
    return selected_features

# %%



