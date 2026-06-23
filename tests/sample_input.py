from src.data_processing import buil_features,load_data




def sample_input():
    df=load_data()
    X , Y = buil_features(df)   
    return X.head()


if __name__=='__main__':
    print(sample_input())