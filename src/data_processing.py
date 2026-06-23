from src.config import *
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# load dataset
def load_data():
    return pd.read_csv(DATA_PATH) 
# buil features
def buil_features(df):
    X=df.drop(columns=['Outcome'],axis=1)
    Y=df['Outcome']
    return X,Y

def buil_preprocessor_regression(X):  
    num_cols=X.select_dtypes(include=['int64','float64']).columns.tolist()  
    processor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols)
        ]
    )
    return processor

def get_train_test_data():
    df =load_data()
    X, Y = buil_features(df)
    X_train,X_test,Y_train,Y_test= train_test_split(X, Y , stratify=Y,random_state=RANDOM_STATE,test_size=TEST_SIZE)
    return X_train,X_test,Y_train,Y_test



if __name__=='__main__':
    df =load_data()
    x, y = buil_features(df)
    print(x.head(3))
    print(y.head(5))

