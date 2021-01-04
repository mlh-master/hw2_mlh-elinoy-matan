from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_evel(model,X_test,Y_test):

    plot_confusion_matrix(model, X_test, Y_test, cmap=plt.cm.Blues)
    plt.grid(False)
    plt.show()


    y_pred_test=model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
    TN = calc_TN(Y_test, y_pred_test)
    FP = calc_FP(Y_test, y_pred_test)
    FN = calc_FN(Y_test, y_pred_test)
    TP = calc_TP(Y_test, y_pred_test)
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * Se * PPV) / (Se + PPV)
    print('Sensitivity is {:.2f}. Specificity is {:.2f}. PPV is {:.2f}. NPV is {:.2f}. Accuracy is {:.2f}. F1 is {:.2f}. '.format(
            Se, Sp, PPV, NPV, Acc, F1))
    print('AUROC is {:.2f}'.format(roc_auc_score(Y_test, y_pred_proba_test[:, 1])))

