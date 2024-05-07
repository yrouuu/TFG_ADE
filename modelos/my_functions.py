import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score
    
    
def metrics(y_train, y_pred_train, y_test, y_pred_test):
    np.random.seed(0)
    traning_scores = []
    testing_scores = []
    
    ### Metrics for the traning set ###
    traning_scores.append(accuracy_score(y_train, y_pred_train))
    traning_scores.append(precision_score(y_train, y_pred_train))
    traning_scores.append(recall_score(y_train, y_pred_train))
    traning_scores.append(f1_score(y_train, y_pred_train))
    print('Training Metrics:')
    print(f'Training Accuracy: ', traning_scores[0]) 
    print(f'Training Precision: ',traning_scores[1])
    print(f'Training Recall: ',traning_scores[2])
    print(f'Training F1-Score: ',traning_scores[3])
    print('\n')
    
    ### Metrics for the testing set ###
    testing_scores.append(accuracy_score(y_test, y_pred_test))
    testing_scores.append(precision_score(y_test, y_pred_test))
    testing_scores.append(recall_score(y_test, y_pred_test))
    testing_scores.append(f1_score(y_test, y_pred_test))
    print('Testing Metrics:')
    print(f'Testing Accuracy: ', testing_scores[0]) 
    print(f'Testing Precision: ',testing_scores[1])
    print(f'Testing Recall: ',testing_scores[2])
    print(f'Testing F1-Score: ',testing_scores[3])
    print('\n')
    
    return traning_scores, testing_scores
    
def print_metric_plots(X,y):
    np.random.seed(0)
    # Define a range of train/test sizes
    test_sizes = np.arange(0.1, 1, 0.1) # Test sizes from 10% to 90%

    # Lists to store results
    train_precision = []
    test_precision = []
    train_recall = []
    test_recall = []
    train_accuracy = []
    test_accuracy = []
    train_f1 = []
    test_f1 = []

    # Iterate over different train/test sizes
    for size in test_sizes:
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
        sc= StandardScaler()
        X_train_rescaled = sc.fit_transform(X_train)
        X_test_rescaled = sc.transform(X_test)
        
        # Train model
        model = LogisticRegression(penalty='l2', C=0.001, random_state=42,solver='lbfgs')
        model.fit(X_train_rescaled, y_train)
        
        # Predictions
        #train_preds = model.predict(X_train_rescaled)
        test_preds = model.predict(X_test_rescaled)
        
        # Calculate scores and append results to lists
        #train_accuracy.append(accuracy_score(y_train, train_preds))
        test_accuracy.append(accuracy_score(y_test, test_preds))
        #train_precision.append(precision_score(y_train, train_preds))
        test_precision.append(precision_score(y_test, test_preds))
        #train_recall.append(recall_score(y_train, train_preds))
        test_recall.append(recall_score(y_test, test_preds))
        #train_f1.append(f1_score(y_train, train_preds))
        test_f1.append(f1_score(y_test, test_preds))
    
    #plt.plot(test_sizes, train_accuracy, label='Train Accuracy',color='#ffb482')
    plt.plot(test_sizes, test_accuracy, label='Test Accuracy',color='#a1c9f4')
    plt.xlabel('Model Test Size (%)')
    plt.ylabel('Accuracy Score')
    plt.title('Effect of Test Size on model Accuracy',fontweight = 'bold')
    plt.legend()
    plt.show()
    
    #plt.plot(test_sizes, train_precision, label='Train Precision', color='#ffb482')
    plt.plot(test_sizes, test_precision, label='Test Precision', color='#a1c9f4')
    plt.xlabel('Model Test Size (%)')
    plt.ylabel('Precision Score')
    plt.title('Effect of Test Size on model Precision',fontweight = 'bold')
    plt.legend()
    plt.show()
    
    #plt.plot(test_sizes, train_recall, label='Train Recall',color='#ffb482')
    plt.plot(test_sizes, test_recall, label='Test Recall',color='#a1c9f4')
    plt.xlabel('Model Test Size (%)')
    plt.ylabel('Recall Score')
    plt.title('Effect of Test Size on model Recall',fontweight = 'bold')
    plt.legend()
    plt.show()
    
    #plt.plot(test_sizes, train_f1, label='Train F1-Score',color='#ffb482')
    plt.plot(test_sizes, test_f1, label='Test F1-Score',color='#a1c9f4')
    plt.xlabel('Model Test Size (%)')
    plt.ylabel('F1-Score')
    plt.title('Effect of Test Size on model F1-Score',fontweight = 'bold')
    plt.legend()
    plt.show()

def roc(X,y):
    
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc= StandardScaler()
    X_train_rescaled = sc.fit_transform(X_train)
    X_test_rescaled = sc.transform(X_test)
    
    # Train model
    model = LogisticRegression(penalty='l2', C=0.001, random_state=42,solver='lbfgs')
    model.fit(X_train_rescaled, y_train)
    
    #train_proba_preds = model.predict_proba(X_train_rescaled)[:, 1]
    test_proba_preds = model.predict_proba(X_test_rescaled)[:, 1]
    
    #fpr, tpr, thresholds = roc_curve(y_train, train_proba_preds)
    #auc_train = roc_auc_score(y_train, train_proba_preds)
    
    # Plot ROC curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#ff9f9b', label='ROC Curve (AUC = {:.2f})'.format(auc_train))
    plt.plot([0, 1], [0, 1], 'k--',color='#aaa')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Training Set',fontweight = 'bold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()"""
    
    fpr, tpr, thresholds = roc_curve(y_test, test_proba_preds)
    auc_test = roc_auc_score(y_test, test_proba_preds)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#ff9f9b', label='ROC Curve (AUC = {:.2f})'.format(auc_test))
    plt.plot([0, 1], [0, 1],'k--',color='#aaa')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Test Set',fontweight = 'bold')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()