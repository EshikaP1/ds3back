## Python Titanic Model
# Import the required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import numpy as np

# Define the TitanicRegression global variable
titanic_regression = None

# Define the TitanicRegression class
class TitanicRegression:
    def __init__(self):
        self.dt = None
        self.logreg = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None
        self.datacolumns = None
    
    def inittitanic(self):
        import seaborn as sns
        # Load the titanic dataset
        titanic_data = sns.load_dataset('titanic')
        td = titanic_data
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)

        # Encode categorical variables
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(td[['embarked']])
        onehot = enc.transform(td[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        td[cols] = pd.DataFrame(onehot)
        td.drop(['embarked'], axis=1, inplace=True)
        td.dropna(inplace=True) # drop rows with at least one missing value, after preparing the data
    
        #titanic_data = sns.load_dataset('titanic')
        #X = titanic_data.drop('survived', axis=1)
        #y = titanic_data['survived']
        
        X = td.drop('survived', axis = 1)
        y = td['survived']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.datacolumns = self.X_train.columns
        
        print("Training data columns")
        print(self.datacolumns)

        # Initialize the encoder
        #self.encoder = OneHotEncoder(handle_unknown='ignore')
        #self.X_train = self.encoder.fit_transform(self.X_train)
        #self.X_test = self.encoder.transform(self.X_test)

        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X_train, self.y_train)

        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)

    def runDecisionTree(self):
        if self.dt is None:
            print("Decision Tree model is not initialized. Please run initTitanic() first.")
            return
        y_pred_dt = self.dt.predict(self.X_test)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        print('Decision Tree Classifier Accuracy: {:.2%}'.format(accuracy_dt))

    def runLogisticRegression(self):
        if self.logreg is None:
            print("Logistic Regression model is not initialized. Please run initTitanic() first.")
            return
        y_pred_logreg = self.logreg.predict(self.X_test)
        accuracy_logreg = accuracy_score(self.y_test, y_pred_logreg)
        print('Logistic Regression Accuracy: {:.2%}'.format(accuracy_logreg))
        
        
def predictSurvival(passenger):
    
    passenger_df = pd.DataFrame(passenger, index=[0])
    print(passenger_df)   
    passenger_df.drop(['name'], axis=1, inplace=True)
    passengerToPredict = passenger_df.copy()
    
    passengerToPredict['sex'] = passengerToPredict['sex'].apply(lambda x: 1 if x == 'male' else 0)
    passengerToPredict['alone'] = passengerToPredict['alone'].apply(lambda x: 1 if x == True else 0)
    
    # Encode categorical variables
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(passengerToPredict[['embarked']])
    onehot = enc.transform(passengerToPredict[['embarked']]).toarray()
    cols = ['embarked_' + val for val in enc.categories_[0]]
    passengerToPredict[cols] = pd.DataFrame(onehot, index=passengerToPredict.index)
    passengerToPredict.drop(['embarked'], axis=1, inplace=True)
    passengerToPredict.dropna(inplace=True) # drop rows with at least one missing value, after preparing the data
    
    # Add missing columns and fill them with default values
    missing_cols = set(titanic_regression.datacolumns) - set(passengerToPredict.columns)
    for col in missing_cols:
        passengerToPredict[col] = 0
        
        
    
    #enc = OneHotEncoder(handle_unknown='ignore')
    #logreg = LogisticRegression()
    #passenger_df = pd.DataFrame(passenger, index=[0])  
    #passenger = passenger_df.copy()
    #new_passenger = passenger.copy()
    
    # Preprocess the new passenger data
    #new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
    #new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

    # Encode 'embarked' variable
    #titanic_regression.encoder.fit(new_passenger[['embarked']])
    #onehot = titanic_regression.encoder.transform(new_passenger[['embarked']]).toarray()
    #cols = ['embarked_' + val for val in titanic_regression.encoder.categories_[0]]
    #new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
    #new_passenger.drop(['name'], axis=1, inplace=True)
    #new_passenger.drop(['embarked'], axis=1, inplace=True)


    # Predict the survival probability for the new passenger
    #dead_proba, alive_proba = np.squeeze(titanic_regression.logreg.predict_proba(new_passenger))

    # Print the survival probability
    #print('Death probability: {:.2%}'.format(dead_proba))  

    #print('Survival probability: {:.2%}'.format(alive_proba))

    # Add missing columns and fill them with default values
    #print(titanic_regression.X_train)
    #missing_cols = set(titanic_regression.datacolumns) - set(new_passenger.columns)
    #for col in missing_cols:
    #    new_passenger[col] = 0

    # Ensure the order of column in the passenger matches the order in the training data
    #new_passenger = new_passenger[titanic_regression.datacolumns]

    # Preprocess the passenger data
    #new_passenger = titanic_regression.encoder.transform(new_passenger)
    
    passengerToPredict = passengerToPredict[titanic_regression.datacolumns]
    
    print(passengerToPredict)
    predict = titanic_regression.logreg.predict(passengerToPredict)
    return predict

def initTitanic():
    global titanic_regression
    titanic_regression = TitanicRegression()
    titanic_regression.inittitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()
    #titanic_regression.predictSurvival()

