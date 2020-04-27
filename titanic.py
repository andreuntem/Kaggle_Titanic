# ===========================================================================
# KAGGLE TITANIC FUNCTIONS
# ===========================================================================

# Import libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ===========================================================================
# FUNCTIONS: IMPORT DATA & FEATURE ENGINEERING
# ===========================================================================

def import_data(filename):
    df = pd.read_csv(filename, index_col='PassengerId')
    return df


def concatenate_dataframes(df1, df2):
    df = pd.concat([df1, df2], sort=False)
    return df


def assign_grouped_title(x):
    military = [' Capt', ' Col', ' Major']
    don_donna = [' Don', ' Dona']
    dr = [' Dr']
    noble = [' Jonkheer', ' Sir', ' the Countess']
    master = [' Master']
    miss = [' Miss']
    mme_mlle = [' Mme', ' Mlle']
    mr = [' Mr']
    mrs_ms_lady = [' Mrs', ' Ms', ' Lady']
    rev = [' Rev']
    if x in military:
        return 'military'
    elif x in don_donna:
        return 'don_donna'
    elif x in dr:
        return 'dr'
    elif x in noble:
        return 'noble'
    elif x in master:
        return 'master'
    elif x in miss:
        return 'miss'
    elif x in mme_mlle:
        return 'mme_mlle'
    elif x in mr:
        return 'mr'
    elif x in mrs_ms_lady:
        return 'mrs_ms_lady'
    elif x in rev:
        return 'rev'


def assign_group_title_class(title_, class_):
    if title_ in ['mr', 'mrs_ms_lady', 'miss', 'master']:
        return title_+'_'+class_
    else:
        return title_


def create_features(df, standardize=False):
    
    # Extract Title from name
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0])
    # df.drop('Name', axis=1, inplace=True)

    # Group Title by low/high survival
    title_low_survival = [' Capt', ' Col', ' Don', ' Dona', ' Dr', ' Jonkheer', ' Major', ' Mr', ' Rev']
    df['Survival_by_title'] = df['Title'].apply(lambda x: 'high' if x not in title_low_survival else 'low')

    # Granular Grouped  Title
    df['granular_grouped_title'] = df['Title'].apply(assign_grouped_title)

    # Grouped Title 'Mr, Mrs, Miss' & Pclass
    df['title_class'] = df.apply(lambda x: assign_group_title_class(x['granular_grouped_title'], str(x['Pclass'])), axis=1)

    # Average fare per person
    avg_fare = df[['Ticket', 'Fare']].groupby(by='Ticket').mean()
    counter = Counter(df['Ticket'])
    passenger_ticket = pd.DataFrame.from_dict(counter, orient='index', columns=['n_passenger'])
    avg_fare = avg_fare.join(passenger_ticket, how='left')
    avg_fare['avg_fare'] = avg_fare['Fare']/avg_fare['n_passenger']
    df = pd.merge(df, avg_fare[['avg_fare']], how='left', left_on='Ticket', right_on='Ticket')
    df.drop(['Fare', 'Ticket'], axis=1, inplace=True)

    # Mean age by class and sex
    sex_class_age = df[['Pclass','Sex','Age']].copy()
    sex_class_age['Pclass_Sex'] = sex_class_age['Pclass'].apply(str) + '_' + sex_class_age['Sex']
    avg_age = sex_class_age.groupby(by='Pclass_Sex').mean()[['Age']]
    df['Pclass_Sex'] = df['Pclass'].apply(str) + '_' + df['Sex']
    df = pd.merge(df, avg_age, how='left', left_on='Pclass_Sex', right_on='Pclass_Sex')
    df['Age'] = df[['Age_x', 'Age_y']].apply(lambda x: x['Age_x'] if x['Age_x']>0 else x['Age_y'], axis=1)
    df.drop(['Age_x', 'Age_y', 'Pclass_Sex'], axis=1, inplace=True)

    # Median fare per person
    median_fare_class = df[['Pclass', 'avg_fare']].copy()
    median_fare_class = median_fare_class.groupby(by='Pclass').median()
    df = pd.merge(df, median_fare_class, how='left', left_on='Pclass', right_on='Pclass')
    df.drop(['avg_fare_x', 'avg_fare_y'], axis=1, inplace=True)

    if standardize:
        df['Age'] = (df['Age'] - df['Age'].mean())/df['Age'].std()

    return df


def select_features(df, features):
    return df[features]


def convert_categorical(df, variables):
    # return pd.get_dummies(df, columns=variables)
    return pd.get_dummies(df, columns=variables, drop_first=True)


def extract_X_y(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y


# ===========================================================================
# FUNCTIONS: GRID SEARCH: HYPER PARAMETER TUNNING
# ===========================================================================

def train_model(model_type, X_train, y_train):

    if model_type == 'LogisticRegression':
        model = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1)
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class', 'Age']
        parameters = {'C':[.1, .3, .5, .75, 1, 1.25, 1.5, 3]}
    
    if model_type == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier()
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class', 'Age']
        parameters = {'max_depth':[3, 4, 5, 6, 7, 8, 9, 10]}

    if model_type == 'KNeighborsClassifier':
        model = KNeighborsClassifier(n_jobs=-1)
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        parameters = {'n_neighbors':[25, 30, 35, 40, 45, 50]}

    if model_type == 'SVC':
        model = SVC()
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        parameters = {'C':[.001, .005, .01, .05, .1, 1, 10]}

    if model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(n_jobs=-1)
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        parameters = {'max_depth':[5, 7, 10], 'n_estimators':[250, 300, 350]}

    if model_type == 'XGBoost':
        model = XGBClassifier(n_jobs=-1)
        # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Title', 'title_class']
        parameters = {'learning_rate':[.01, .05, .1], 'n_estimators':[100, 200, 300]}

    gridsearch = GridSearchCV(estimator=model, param_grid=parameters, cv=20, scoring='accuracy')
    gridsearch.fit(X_train, y_train)
    return gridsearch


def predict(model, X_eval, df, column='Survived'):
    indexes = X_eval.index
    y_pred = model.predict(X_eval)
    if column not in df.columns:
        df[column] = np.nan
    df[column][indexes] = y_pred
    return df


def gridsearch_output(gridsearch):
    gs = pd.DataFrame(gridsearch.cv_results_)
    gs.sort_values(by='mean_test_score', ascending=False, inplace=True)
    gs.reset_index(inplace=True)
    return gs[['params', 'mean_test_score', 'std_test_score']]


# ===========================================================================
# FUNCTIONS: MODEL SELECTION
# ===========================================================================

def fit_predict_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
