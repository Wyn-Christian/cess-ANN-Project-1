# ARTIFICIAL NEURAL NETWORK TEMPLATE

# IMPORT THE LIBRARIES
import numpy as np # for numerical computation
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for data visualization

# PART 1. DATA PREPROCESSING

# A. IMPORT THE DATASET
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:,13].values

# For the Data Information
dataset.info()

# B. Encoding the Categorical Data

# B.1 One-hot Encoding the "Geography" to Create Dummy Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_transformer = ColumnTransformer([("Geography", OneHotEncoder(categories = "auto"), [1])], remainder="passthrough")

X = column_transformer.fit_transform(X)

#B.2 Label Encoding the "Gender" Column to convert it to Numeric
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X[:,4] = label_encoder.fit_transform(X[:,4])
X = X.astype(float)

# C. Splitting the Dataset into Training Dataset & Testing Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0, stratify=None)


# D. Perform Feature Scaling
from sklearn.preprocessing import StandardScaler

X_train_standard = X_train.copy()
X_test_standard = X_test.copy()

standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train_standard)
X_test_standard = standard_scaler.transform(X_test_standard)


# PART 2. BUILDING THE ARTIFICIAL NEURAL NETWORK MODEL

# A. Importing the Keras Libraries and Packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# B. Initializing the ANN
classifier = Sequential()

# C. Adding the Input Layer and the First Hidden Layer
classifier.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 12 ))
classifier.add(Dropout(rate = 0.1))

# D. Adding the Second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.1))

# E. Adding the Output Layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))


# PART 3. TRAINING THE ARTIFICIAL NEURAL NETWORK MODEL

# A. Compiling the ANN 
classifier.compile(optimizer = 'sgd', loss = "binary_crossentropy", metrics = ["accuracy"])

# B. Fitting the ANN Model on the Training Dataset
classifier.fit(x = X_train_standard, y = Y_train, batch_size = 50, epochs = 50)

# C. Summarizing the ANN Model
classifier.summary()

# PART 4. MAKING PREDICTIONS AND EVALUATING THE ANN MODEL

# A. Predicting the Output of the Testing Dataset

Y_predict_probability = classifier.predict(X_test_standard)
Y_predict = np.rint(Y_predict_probability)

# B. To Generate and Plot the Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y_test, Y_predict)

import seaborn as sns
plt.figure(figsize = (10, 7))
sns.heatmap(confusion_matrix, annot = True)
plt.xlabel("Predicted Value")
plt.ylabel("Expected Value")

# C. Computing the Hold-Out Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_predict)
print("Hold-out Accuracy")
print(accuracy)
print("  ")

# D. Generating the Classification Report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(Y_test, Y_predict))
print("  ")


# E. Predicting the Output of the Single Observation

# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 Years Old
# Tenure: 3 Years
# Balance: $ 60,000
# Number of Products: 2
# With Credit Card: Yes
# Active Member: Yes
# Estimated Salary: $ 50,000

# SOLUTION
# sample = [[1,0,0,600,1,40,3,60000,2,1,1,50000]]
# sample_standard = sample.copy()
# sample_standard = standard_scaler.transform(sample_standard)
# sample_predict = classifier.predict(sample_standard)
# Y_predict = np.rint(Y_predict_probability)
# print(Y_predict)





# PART 5: PERFORM K-FOLD CROSS-VALIDATION TO ASSESS THE PERFORMANCE OF THE ANN MODEL

# A. To Feature Scale the X Variable Using the StandardScaler
X_standard = X.copy()
X_standard = standard_scaler.fit_transform(X_standard)

# B. Build the ANN Classifier Using the KerasClassifier
from scikeras.wrappers import KerasClassifier 

def classifier(): 
    classifier = Sequential()
    classifier.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 12 ))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu"))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
    classifier.compile(optimizer = 'sgd', loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

ann_model = KerasClassifier(model = classifier, batch_size = 10, epochs = 50)

# C. Import the StratifiedKFold Class
from sklearn.model_selection import StratifiedKFold

k_fold = StratifiedKFold(n_splits = 10, shuffle = False, random_state = None)

# D. Import the cross_val_score Class
from sklearn.model_selection import cross_val_score

# Try the following Performance Metrics
    # A. accuracy, "accuracy"
    # B. precision, "precision"
    # C. recall, "recall"
    # D. F1-score, "f1"
    # E. ROC-AUC, "roc_auc"

# E. Perform the k-Fold Cross Validation

    # E.1 Using Accuracy as the Scoring Metric for Cross-Valdiation
accuracies = cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "accuracy", n_jobs = 1)
accuracies_average = accuracies.mean()
accuracies_deviation = accuracies.std()
    
print ("Accuracies of k-folds:")
print (accuracies)
print ("  ")
print ("Average Accuracy of k-folds:")
print (accuracies_average)
print ("  ")
print ("Accuracy Deviation of k-folds:")
print (accuracies_deviation)
print ("  ")    
  

    # E.2 Using F1-Score as the Scoring Metric for Cross-Valdiation
f1 = cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "f1", n_jobs = 1)
f1_average = f1.mean()
f1_deviation = f1.std()
    
print ("F1-Scores of k-folds:")
print (f1)
print ("  ")
print ("Average F1-Score of k-folds:")
print (f1_average)
print ("  ")
print ("F1-Score Deviation of k-folds:")
print (f1_deviation)
print ("  ")    



    # E.3 Using Precision as the Scoring Metric for Cross-Valdiation
precision = cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "precision", n_jobs = 1)
precision_average = precision.mean()
precision_deviation = precision.std()
    
print ("Precision of k-folds:")
print (precision)
print ("  ")
print ("Average Precision of k-folds:")
print (precision_average)
print ("  ")
print ("Precision Deviation of k-folds:")
print (precision_deviation)
print ("  ")    


    # E.4 Using Recall as the Scoring Metric for Cross-Valdiation
recall = cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "recall", n_jobs = 1)
recall_average = recall.mean()
recall_deviation = recall.std()
    
print ("Recall of k-folds:")
print (recall)
print ("  ")
print ("Average Recall of k-folds:")
print (recall_average)
print ("  ")
print ("Recall Deviation of k-folds:")
print (recall_deviation)
print ("  ")   


    # E.5 Using ROC-AUC as the Scoring Metric for Cross-Valdiation
roc_auc = cross_val_score(estimator = ann_model, X = X_standard, y = Y, cv = k_fold, scoring = "roc_auc", n_jobs = 1)
roc_auc_average = roc_auc.mean()
roc_auc_deviation = roc_auc.std()
    
print ("ROC-AUC of k-folds:")
print (roc_auc)
print ("  ")
print ("Average ROC-AUC of k-folds:")
print (roc_auc_average)
print ("  ")
print ("ROC-AUC Deviation of k-folds:")
print (roc_auc_deviation)
print ("  ")   


# PART 6. PERFORM HOLD-OUT VALIDATION TO ASSESS THE ARTIFICIAL NEURAL NETWORK MODEL'S PERFORMANCE

TP = confusion_matrix[1, 0]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]


# A. For Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict)
print("Classification Accuracy: %.4f"
      %classification_accuracy)
print (" ")


# B. For Classification Error
classification_error = 1 - classification_accuracy
print("Classification Error: %.4f"
      %classification_error)
print (" ")


# C. For the Sensitivity, Recall Score, Probability of Detection, True Positive Rate
from sklearn.metrics import recall_score
recall = recall_score(Y_test, Y_predict, average = "weighted")
print("Recall Score: %.4f"
      %recall)
print (" ")


# D. For the Specificity or True Negative Rate
Specificity = TN / (TN + FP)
print("Specificity: %.4f"
      %Specificity)
print (" ")


# E. For the False Positive Rates
false_positive_rate = 1 - Specificity
print("False Positive Rate: %.4f"
      %false_positive_rate)
print (" ")


# F. For the Precision or Positive Predictive Value
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict, average = "weighted")
print("Precision Score: %.4f"
      %precision)
print (" ")


# G. For the F1-Score
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict, average = "weighted")
print("F1-Score: %.4f"
      %f1_score)
print (" ")


# H. For the Classification Report
from sklearn.metrics import classification_report
classification_report = classification_report(Y_test, Y_predict)
print (classification_report)


# I. For the Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision_value, recall_value, threshold = precision_recall_curve(Y_test, Y_predict)

plt.plot(precision_value, recall_value)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title ("Precision Recall Curve for the ANN Model")
plt.xlabel ("Precision")
plt.ylabel("Recall")


# J. For the ROC Curve with AUC

# J.1 For the Receiver Operating Curve (ROC)
from sklearn.metrics import roc_curve
FPR, TPR, threshold = roc_curve(Y_test, Y_predict)

# J.2 For the Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score
AUC_score = roc_auc_score(Y_test, Y_predict)

# J.3 To Plot the ROC Curve with AUC
plt.plot(FPR, TPR, label = "ROC Curve")
plt.xlim([0, 1])
plt.ylim([0, 1])

# J.4 For the Plot of Baseline for AUC
plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label = "Baseline", linestyle = "--")
plt.title (f"ROC Curve with AUC = {round(AUC_score, 4)}), for the ANN Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


# PART 7. PERFORM HYPERPARAMETER TUNING TO OPTIMIZE THE ARTIFICIAL NEURAL NETWORK MODEL

# A. Tune First the Batch Size and Epochs

# A.1 Build the ANN Model for the Optimization Process
def classifier_optimization(optimizer = "sgd"):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 12 )) 
    classifier_optimization.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu" ))
    classifier_optimization.add(Dense(units = 1 , kernel_initializer = "glorot_uniform", activation = "sigmoid" ))
    classifier_optimization.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier_optimization


ann_model_optimization = KerasClassifier(model = classifier_optimization)

# A.2 To Import gridSearchCV Class and Optimize the Parameter

# To Import the GridSearchCV
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized for the ANN Model
parameters = {"batch_size": [50, 100, 200, 500],
              "epochs": [100, 150, 200, 250]}

# To Define the GridSearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv = k_fold,
                           n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print("  ")

print ("Best Parameters")
print(best_parameters)
print("  ")


# B. Tune Next the Optimizer

# B.1 Build the ANN Model for the Optimization Process

def classifier_optimization(optimizer = "sgd"):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 12 )) 
    classifier_optimization.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu" ))
    classifier_optimization.add(Dense(units = 1 , kernel_initializer = "glorot_uniform", activation = "sigmoid" ))
    classifier_optimization.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier_optimization


ann_model_optimization = KerasClassifier(model = classifier_optimization, batch_size = 50, epochs = 200)

# B.2 To Import gridSearchCV Class and Optimize the Parameter

# To Import the GridSearchCV
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized for the ANN Model
# parameters =  {'optimizer': ['adam', 'sgd', 'rmsprop', 'adamW', 'adadelta', 'adagrad', 'adamax', 'adafactor', 'aadam', 'ftrl', 'Lion', 'LossScaleOptimizer']}

parameters =  {'optimizer': ['adam', 'sgd', 'rmsprop', 'adamW', 'ftrl', ]}

# To Define the GridSearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv = k_fold,
                           n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print("  ")

print ("Best Parameters")
print(best_parameters)
print("  ")



# C. Tune Next the Optimizer's Learning Rate and Momentum


# C.1 Build the ANN Model for the Optimization Process

from keras.optimizers import RMSprop

def classifier_optimization(learning_rate, momentum):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 12 )) 
    classifier_optimization.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu" ))
    classifier_optimization.add(Dense(units = 1 , kernel_initializer = "glorot_uniform", activation = "sigmoid" ))
    optimizer_setting = RMSprop(learning_rate = learning_rate, momentum = momentum)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier_optimization


ann_model_optimization = KerasClassifier(model = classifier_optimization, learning_rate = 0.001, momentum = 0.0, batch_size = 50, epochs = 200)

# C.2 To Import gridSearchCV Class and Optimize the Parameter

# To Import the GridSearchCV
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized for the ANN Model
parameters =  {'learning_rate': [0.001, 0.01, 0.1, 1.0], 'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]}

# To Define the GridSearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv = k_fold,
                           n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print("  ")

print ("Best Parameters")
print(best_parameters)
print("  ")


#Best Parameters {'batch_size': 50, 'epochs':200}
#Best Parameters {'optimizer': rmsprop'}
#Best Parameters {'learning_rate': 0.001, 'momentum':0.2}



# D. Tune Next the Network's Weight Initialization


# D.1 Build the ANN Model for the Optimization Process

def classifier_optimization(kernel_initializer):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = kernel_initializer, activation = "relu", input_dim = 12 )) 
    classifier_optimization.add(Dense(units = 6, kernel_initializer = kernel_initializer, activation = "relu" ))
    classifier_optimization.add(Dense(units = 1 , kernel_initializer = kernel_initializer, activation = "sigmoid" ))
    optimizer_setting = RMSprop(learning_rate = 0.001, momentum = 0.2)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier_optimization


ann_model_optimization = KerasClassifier(model = classifier_optimization, kernel_initializer = "glorot_uniform", batch_size = 50, epochs = 200)

# D.2 To Import gridSearchCV Class and Optimize the Parameter

# To Import the GridSearchCV
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized for the ANN Model
# parameters = {'kernel_initializer': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']}
parameters = {'kernel_initializer': ['uniform', 'normal', 'zero', 'glorot_uniform']}

# To Define the GridSearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv = k_fold,
                           n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print("  ")

print ("Best Parameters")
print(best_parameters)
print("  ")

#Best Parameters {'batch_size': 50, 'epochs':200}
#Best Parameters {'optimizer': rmsprop'}
#Best Parameters {'learning_rate': 0.001, 'momentum':0.2}
#Best Parameters {"kernel_initializer': glorot_uniform}


# E. Tune Next the Neuron Activation Function


# E.1 Build the ANN Model for the Optimization Process

def classifier_optimization(activation1, activation2, activation3):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer = "glorot_uniform", activation = activation1, input_dim = 12 )) 
    classifier_optimization.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = activation2 ))
    classifier_optimization.add(Dense(units = 1 , kernel_initializer = "glorot_uniform", activation = activation3 ))
    optimizer_setting = RMSprop(learning_rate = 0.001, momentum = 0.2)
    classifier_optimization.compile(optimizer = optimizer_setting, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier_optimization


ann_model_optimization = KerasClassifier(model = classifier_optimization, activation1 = "relu", activation2 = "relu", activation3 = "relu", batch_size = 50, epochs = 200)

# E.2 To Import gridSearchCV Class and Optimize the Parameter
#parameters = {'activation1': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#              'activation2': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#              'activation3': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']}

parameters = {'activation1': ['softmax', 'relu', 'tanh', 'sigmoid'],
              'activation2': ['softmax', 'relu', 'tanh', 'sigmoid'],
              'activation3': ['softmax', 'relu', 'tanh', 'sigmoid']}


# To Import the GridSearchCV
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized for the ANN Model


# To Define the GridSearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, 
                           param_grid = parameters, 
                           scoring = "accuracy",
                           cv = k_fold,
                           n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridSearchCV
results = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print("  ")

print ("Best Parameters")
print(best_parameters)
print("  ")

#Best Parameters {'batch_size': 50, 'epochs':200}
#Best Parameters {'optimizer': rmsprop'}
#Best Parameters {'learning_rate': 0.001, 'momentum':0.2}
#Best Parameters {"kernel_initializer': glorot_uniform}
# Best Parameters {'activation1': 'relu', 'activation2': 'sigmoid' , 'activation3': 'sigmoid'} 


# F. Tune Next the Dropout Regularization

# F.1 Build the ANN Model for the Optimizer Process

from keras.constraints import MaxNorm

def classifier_optimization(dropout_rate, weight_constraint):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = 7, kernel_initializer="glorot_uniform", activation = 'relu', kernel_constraint=MaxNorm(weight_constraint), input_dim = 12))
    classifier_optimization.add(Dropout(rate=dropout_rate))
    classifier_optimization.add(Dense(units = 6, kernel_initializer="glorot_uniform", activation = 'relu', kernel_constraint=MaxNorm(weight_constraint)))
    classifier_optimization.add(Dropout(rate=dropout_rate))
    classifier_optimization.add(Dense(units = 1, kernel_initializer="glorot_uniform", activation = 'sigmoid', kernel_constraint=MaxNorm(weight_constraint)))
    classifier_optimization.add(Dropout(rate=dropout_rate))
    optimizer_setting = RMSprop(learning_rate = 0.001, momentum = 0.9)
    classifier_optimization.compile(optimizer = optimizer_setting, loss= "binary_crossentropy", metrics=["accuracy"])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model = classifier_optimization, dropout_rate = 0.0, weight_constraint = 0.5, batch_size = 50, epochs=200)

# F.2 To Import GridSearchCV Class and Optimize the Parameter


# To Import the GridsearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'weight_constraint': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

# To Define the GridsearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, param_grid = parameters, scoring = "accuracy", cv = k_fold, n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridsearchCV
results = pd.DataFrame(grid_search.cv_results_)[["mean_test_score", "std_test_score", "params"]]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print(" ")

print("Best Parameters")
print(best_parameters)
print(" ")

# G. Tune Next the Number of Neurons in the Hidden Layer

# G.1 Build the ANN Model for the Optimizer Process


def classifier_optimization(neuron1, neuron2):
    classifier_optimization = Sequential()
    classifier_optimization.add(Dense(units = neuron1, kernel_initializer="glorot_uniform", activation = 'relu', kernel_constraint=MaxNorm(2.0), input_dim = 12))
    classifier_optimization.add(Dropout(rate=0.0))
    classifier_optimization.add(Dense(units = neuron2, kernel_initializer="glorot_uniform", activation = 'relu', kernel_constraint=MaxNorm(2.0)))
    classifier_optimization.add(Dropout(rate=0.0))
    classifier_optimization.add(Dense(units = 1, kernel_initializer="glorot_uniform", activation = 'sigmoid', kernel_constraint=MaxNorm(2.0)))
    classifier_optimization.add(Dropout(rate=0.0))
    optimizer_setting = RMSprop(learning_rate = 0.001, momentum = 0.9)
    classifier_optimization.compile(optimizer = optimizer_setting, loss= "binary_crossentropy", metrics=["accuracy"])
    return classifier_optimization

ann_model_optimization = KerasClassifier(model = classifier_optimization, neuron1 = 10, neuron2 = 10, batch_size = 50, epochs=200)

# G.2 To Import GridSearchCV Class and Optimize the Parameter


# To Import the GridsearchCV
from sklearn.model_selection import GridSearchCV

parameters = {'neuron1': [10, 15, 20, 25, 30],
              'neuron2': [10, 15, 20, 25, 30]}

# To Define the GridsearchCV
grid_search = GridSearchCV(estimator = ann_model_optimization, param_grid = parameters, scoring = "accuracy", cv = k_fold, n_jobs = 1)
grid_search = grid_search.fit(X_standard, Y)
print(grid_search)

# To View the Results of the GridsearchCV
results = pd.DataFrame(grid_search.cv_results_)[["mean_test_score", "std_test_score", "params"]]
print(results)

# To Identify the Best Accuracy and the Best Parameters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Accuracy Score")
print(best_accuracy)
print(" ")

print("Best Parameters")
print(best_parameters)
print(" ")

#Best Parameters {'batch_size': 50, 'epochs':200}
#Best Parameters {'optimizer': rmsprop'}
#Best Parameters {'learning_rate': 0.001, 'momentum':0.2}
#Best Parameters {"kernel_initializer': glorot_uniform}
# Best Parameters {'activation1': 'relu', 'activation2': 'sigmoid' , 'activation3': 'sigmoid'} 
# Best Parameters {'dropout_rate': 0.0, 'weight_constraint': 2.5}
# Best Parameters {'neuron1': 15, 'neuron2': 10}
# Best Accuracy Score: 0.8632000000000002
