
from src.data_processing import *
from sklearn.metrics import recall_score ,roc_auc_score,f1_score,accuracy_score


X_train,X_test,Y_train,Y_test=get_train_test_data()
 


def evaluate(model,threshold=0.5):   

    proba = model.predict_proba(X_test)[:, 1]
    Y_pred = (proba > threshold).astype(int)

    accuracy=accuracy_score(Y_test,Y_pred)
    recall =recall_score(Y_test,Y_pred)
    f1score=f1_score(Y_test,Y_pred)
    roc_auc=roc_auc_score(Y_test,proba)

    return accuracy , recall,f1score,roc_auc

""" if __name__=='__main__':
    accuracy , recall,f1score,roc_auc=evaluate()

    print(f"accuracy  :{accuracy:4f}")
    print(f"recall  : {recall:4f}")
    print(f"f1score  :{f1score:4f}")
    print(f"roc_auc  :{roc_auc:4f}")
 """
