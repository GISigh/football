# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:07:53 2021

@author: 14062
"""

# Import scraping modules
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# %% Run for loop to pull game results from Pro football reference

# url = 'https://www.pro-football-reference.com/years/2019/games.htm'

# # Open URL and pass to BeautifulSoup
# html = urlopen(url)
# soup = BeautifulSoup(html)

# soup.prettify()

# column_headers = soup.findAll('tr')[0]
# column_headers = [i.getText() for i in column_headers.findAll('th')]

# # Collect table rows
# rows = soup.findAll('tr')[1:]

# # Get stats from each row
# stats = []
# for i in range(len(rows)):
#   stats.append([col.getText() for col in rows[i].findAll('td')])
  
# # Create DataFrame from our scraped data
# df = pd.DataFrame(stats, columns=column_headers[1:])

# df.to_csv("test.csv" , index = False ) #Write out to look at data

# df.columns.values[4] = 'Home' #Name the 4th column which contans whethere it was a home game with @ symbol

# play_off_start = df[df['Date'] == 'Playoffs'].index #Get the rows with the playoff results

# subset = df[df.index > play_off_start[0]] #subset playoff rows

# subset['Y'] = np.where(subset.Home == '@', 0, 1) #set Y-value as 1 or 0. 1 when home team wins.

# team_names = df['Winner/tie'].unique().tolist() #Get list of unique footballe names

# team_names = [i for i in team_names if i] #Sort and remove none

# %% Parse results for every playoff game 2000 - 2020
#Create a dictionary because the website uses mixed indexing, which is a pain. 
team_dict = {'Baltimore Ravens' :'rav',
 'Buffalo Bills': 'buf',
 'Dallas Cowboys':'dal',
 'Detroit Lions':'det',
 'Green Bay Packers':'gnb',
 'Kansas City Chiefs':'kan',
 'Los Angeles Chargers':'sdg',
 'San Diego Chargers':'sdg',
 'Los Angeles Rams':'ram',
 'Minnesota Vikings':'min',
 'New England Patriots':'nwe',
 'New Orleans Saints':'nor',
 'Oakland Raiders':'rai',
 'Philadelphia Eagles':'phi',
 'San Francisco 49ers':'sfo',
 'Seattle Seahawks':'sea',
 'Tennessee Titans':'oti',
 'Tampa Bay Buccaneers':'tam',
 'Indianapolis Colts':'clt',
 'Houston Texans':'htx',
 'Chicago Bears':'chi',
 'Atlanta Falcons':'atl',
 'Cleveland Browns':'cle',
 'Jacksonville Jaguars':'jax',
 'Carolina Panthers':'car',
 'New York Giants':'nyg',
 'Pittsburgh Steelers':'pit',
 'Arizona Cardinals':'crd',
 'Denver Broncos':'den',
 'Washington Redskins':'was',
 'New York Jets':'nyj',
 'Miami Dolphins':'mia',
 'Cincinnati Bengals':'cin', 
 'St. Louis Rams' : 'ram',
 'Las Vegas Raiders' : 'rai', 
 'Washington Football Team' : 'was'}


years = range(2000,2020)

# create an Empty DataFrame object 
playoff_df = pd.DataFrame() 


for year in years: 
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/games.htm'
    print(url)
    # Open URL and pass to BeautifulSoup
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[0]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    
    # Collect table rows
    rows = soup.findAll('tr')[1:]
    
    # Get stats from each row 
    stats = []
    for i in range(len(rows)):
      stats.append([col.getText() for col in rows[i].findAll('td')])
      
    # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df.columns.values[4] = 'Home' #Name the 4th column which contans whether it was a home game with @ symbol

    play_off_start = df[df['Date'] == 'Playoffs'].index #Get the rows with the playoff results
    
    subset = df[df.index > play_off_start[0]] #subset playoff rows
    
    subset['Y'] = np.where(subset.Home == '@', 0, 1) #set Y-value as 1 or 0. 1 when home team wins.
    
    subset['Year'] = year

    playoff_df = playoff_df.append(subset, ignore_index=True)

# %% Compile all the data
'''Here I use the reults from every game to find the team statistics for every team from 2000 - 2020 and appened it 
to the results data frame. I define 1 as a home win. '''

playoffs = playoff_df[['Winner/tie', 'Loser/tie', 'Y', 'Home', 'Year']]

df_ = pd.DataFrame() 

for index, row in playoffs.iterrows():
    home = row['Y']
    
    if home == 1:
        
        winner = row['Winner/tie']
        loser = row['Loser/tie']    # This isn't the actuall winner/loser, it's whether the home team won.
    else: 
        winner = row['Loser/tie']
        loser = row['Winner/tie']
    year = row['Year']
    print(winner, loser)
    #############Let's get the winners info ###########################################################
    winner = team_dict[winner]
    url = 'https://www.pro-football-reference.com/teams/' + str(winner) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    winner_df = df1.join(df2, lsuffix = '_win_team', rsuffix = '_win_opp')  
    
    #################################Let's get the losers info#############################################
    loser = team_dict[loser]
    url = 'https://www.pro-football-reference.com/teams/' + str(loser) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    lose_df = df1.join(df2, lsuffix = '_lose_team', rsuffix = '_lose_opp')  
    
    lose_df = lose_df.reset_index()
    winner_df = winner_df.reset_index()
    
    df = winner_df.join(lose_df, lsuffix = 'x', rsuffix = 'y')  
    
    df_ = df_.append(df, ignore_index=True)
    


df = playoffs.join(df_, lsuffix = 'x', rsuffix = 'y')

removes = ['Winner/tie', 'Loser/tie', 'Year',	'indexx',	'index_win_team',	'index_win_opp', 'indexy', 
           'index_lose_team', 'index_lose_opp', 'Start_win_team', 'Start_win_opp', 'Start_lose_team', 'Start_lose_opp', 
           'Time_win_team', 'Time_win_opp', 'Time_lose_team', 'Time_lose_opp']


df = df.drop(removes,  axis = 1)


df = df.drop(['Home'],  axis = 1)

# %% Standard train and split -- May need to come back to ply with this a little
# df.to_csv("moop.csv", index = False)    

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.Y, test_size=0.33, random_state=42)    

# %% Random forest
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), X_train), reverse=True))

print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

# %% SVM
#Import svm model
from sklearn import svm

#Create a svm Classifier
# clf = svm.SVC(kernel='poly', degree=2) # poly kernel
# clf = svm.SVC(kernel='rbf') # GAUSSIAN Kernel
# clf = svm.SVC(kernel='linear') # linear Kernel
clf = svm.SVC(kernel='sigmoid') # sigmoid Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))


# %% Naive Bayes
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

# %%
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

NN.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = NN.predict(X_test)   

print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
    
print(metrics.classification_report(y_test, y_pred))
    
# %% Decision Tree
from sklearn import tree
# Instantiate with a max depth of 3
tree_model = tree.DecisionTreeClassifier(max_depth=3)
# Fit a decision tree
tree_model = tree_model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = tree_model.predict(X_test) 

print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
    
print(metrics.classification_report(y_test, y_pred))
    
    
# %% KNN
from sklearn.neighbors import KNeighborsClassifier
# instantiate learning model (k = 3)
knn_model = KNeighborsClassifier(n_neighbors=3)
# fit the model
knn_model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn_model.predict(X_test) 

print(metrics.confusion_matrix(y_test, y_pred))
print("Kappa Score:", metrics.cohen_kappa_score(y_test, y_pred))
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
    
print(metrics.classification_report(y_test, y_pred))
    
# %% Create future data
url = 'https://www.pro-football-reference.com/years/2020/games.htm'
print(url)
# Open URL and pass to BeautifulSoup
html = urlopen(url)
soup = BeautifulSoup(html)
column_headers = soup.findAll('tr')[0]
column_headers = [i.getText() for i in column_headers.findAll('th')]

# Collect table rows
rows = soup.findAll('tr')[1:]

# Get stats from each row 
stats = []
for i in range(len(rows)):
  stats.append([col.getText() for col in rows[i].findAll('td')])
  
# Create DataFrame from our scraped data
df = pd.DataFrame(stats, columns=column_headers[1:])

df.columns.values[4] = 'Home' #Name the 4th column which contans whether it was a home game with @ symbol

df = df[-6:]

playoffs = df[['Winner/tie', 'Loser/tie']]

playoffs['Year'] = 2020

df_ = pd.DataFrame() 

for index, row in playoffs.iterrows():
    winner = row['Loser/tie'] # switched to make home team
    loser = row['Winner/tie']
    year = row['Year']
    print(winner, loser)
    #############Let's get the winners info ###########################################################
    winner = team_dict[winner]
    url = 'https://www.pro-football-reference.com/teams/' + str(winner) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    winner_df = df1.join(df2, lsuffix = '_win_team', rsuffix = '_win_opp')  
    
    #################################Let's get the losers info#############################################
    loser = team_dict[loser]
    url = 'https://www.pro-football-reference.com/teams/' + str(loser) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    lose_df = df1.join(df2, lsuffix = '_lose_team', rsuffix = '_lose_opp')  
    
    lose_df = lose_df.reset_index()
    winner_df = winner_df.reset_index()
    
    df = winner_df.join(lose_df, lsuffix = 'x', rsuffix = 'y')  
    
    df_ = df_.append(df, ignore_index=True)
    
df_.columns

removes = ['indexx',	'index_win_team',	'index_win_opp', 'indexy', 
           'index_lose_team', 'index_lose_opp', 'Start_win_team', 'Start_win_opp', 'Start_lose_team', 'Start_lose_opp', 
           'Time_win_team', 'Time_win_opp', 'Time_lose_team', 'Time_lose_opp']


df = df_.drop(removes,  axis = 1)

df = df.drop('Wild_Card', axis = 1)
# %% Starting to predict Wild card
#Predict the response for test dataset
y_pred = gnb.predict(df)

df['Wild_Card'] = y_pred
gnb.predict_proba(df)
f"{gnb.predict_proba(df):.9f}"#Give me some confidence stuff per game
df.shape
X_train.shape


# %% divison round
df_division = pd.read_csv('division.csv')

df_ = pd.DataFrame() 

for index, row in df_division.iterrows():
    winner = row['Home'] # switched to make home team
    loser = row['Away']
    year = row['Year']
    print(winner, loser)
    #############Let's get the winners info ###########################################################
    winner = team_dict[winner]
    url = 'https://www.pro-football-reference.com/teams/' + str(winner) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    winner_df = df1.join(df2, lsuffix = '_win_team', rsuffix = '_win_opp')  
    
    #################################Let's get the losers info#############################################
    loser = team_dict[loser]
    url = 'https://www.pro-football-reference.com/teams/' + str(loser) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    lose_df = df1.join(df2, lsuffix = '_lose_team', rsuffix = '_lose_opp')  
    
    lose_df = lose_df.reset_index()
    winner_df = winner_df.reset_index()
    
    df = winner_df.join(lose_df, lsuffix = 'x', rsuffix = 'y')  
    
    df_ = df_.append(df, ignore_index=True)
    
df_.columns

removes = ['indexx',	'index_win_team',	'index_win_opp', 'indexy', 
           'index_lose_team', 'index_lose_opp', 'Start_win_team', 'Start_win_opp', 'Start_lose_team', 'Start_lose_opp', 
           'Time_win_team', 'Time_win_opp', 'Time_lose_team', 'Time_lose_opp']


df = df_.drop(removes,  axis = 1)

y_pred = gnb.predict(df)

# df['Wild_Card'] = y_pred

# %% divison round
df_division = pd.read_csv('confrence.csv')

df_ = pd.DataFrame() 

for index, row in df_division.iterrows():
    winner = row['Home'] # switched to make home team
    loser = row['Away']
    year = row['Year']
    print(winner, loser)
    #############Let's get the winners info ###########################################################
    winner = team_dict[winner]
    url = 'https://www.pro-football-reference.com/teams/' + str(winner) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    winner_df = df1.join(df2, lsuffix = '_win_team', rsuffix = '_win_opp')  
    
    #################################Let's get the losers info#############################################
    loser = team_dict[loser]
    url = 'https://www.pro-football-reference.com/teams/' + str(loser) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    lose_df = df1.join(df2, lsuffix = '_lose_team', rsuffix = '_lose_opp')  
    
    lose_df = lose_df.reset_index()
    winner_df = winner_df.reset_index()
    
    df = winner_df.join(lose_df, lsuffix = 'x', rsuffix = 'y')  
    
    df_ = df_.append(df, ignore_index=True)
    
df_.columns

removes = ['indexx',	'index_win_team',	'index_win_opp', 'indexy', 
           'index_lose_team', 'index_lose_opp', 'Start_win_team', 'Start_win_opp', 'Start_lose_team', 'Start_lose_opp', 
           'Time_win_team', 'Time_win_opp', 'Time_lose_team', 'Time_lose_opp']


df = df_.drop(removes,  axis = 1)

y_pred = gnb.predict(df)

# %% divison round
df_division = pd.read_csv('superbowl.csv')

df_ = pd.DataFrame() 

for index, row in df_division.iterrows():
    winner = row['Home'] # switched to make home team
    loser = row['Away']
    year = row['Year']
    print(winner, loser)
    #############Let's get the winners info ###########################################################
    winner = team_dict[winner]
    url = 'https://www.pro-football-reference.com/teams/' + str(winner) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    winner_df = df1.join(df2, lsuffix = '_win_team', rsuffix = '_win_opp')  
    
    #################################Let's get the losers info#############################################
    loser = team_dict[loser]
    url = 'https://www.pro-football-reference.com/teams/' + str(loser) + '/' + str(year) + '.htm'
    html = urlopen(url)
    soup = BeautifulSoup(html)
    column_headers = soup.findAll('tr')[1]
    column_headers = [i.getText() for i in column_headers.findAll('th')]
    rows = soup.findAll('tr')[2:4]
    stats = []
    for i in range(len(rows)):
        stats.append([col.getText() for col in rows[i].findAll('td')])
        # Create DataFrame from our scraped data
    df = pd.DataFrame(stats, columns=column_headers[1:])
    
    df1 = pd.DataFrame(df.iloc[0]).T
    df1 = df1.reset_index()
    df2 = pd.DataFrame(df.iloc[1]).T
    df2 = df2.reset_index()
    
    lose_df = df1.join(df2, lsuffix = '_lose_team', rsuffix = '_lose_opp')  
    
    lose_df = lose_df.reset_index()
    winner_df = winner_df.reset_index()
    
    df = winner_df.join(lose_df, lsuffix = 'x', rsuffix = 'y')  
    
    df_ = df_.append(df, ignore_index=True)
    
df_.columns

removes = ['indexx',	'index_win_team',	'index_win_opp', 'indexy', 
           'index_lose_team', 'index_lose_opp', 'Start_win_team', 'Start_win_opp', 'Start_lose_team', 'Start_lose_opp', 
           'Time_win_team', 'Time_win_opp', 'Time_lose_team', 'Time_lose_opp']


df = df_.drop(removes,  axis = 1)

y_pred = gnb.predict(df)
