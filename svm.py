# -------------------------------------------------------------------------
# AUTHOR: Gina Martinez
# FILENAME: svm.py
# SPECIFICATION: Obtain the highest accuracy and corresponding hyperparameters
# FOR: CS 4200- Assignment #3
# TIME SPENT: 4 hours
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
highest_c = 0
highest_degree = 0
highest_kernel = ''
highest_dfs = ''

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        X_training.append(row[:-1])
        Y_training.append(row[-1])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
# --> add your Python code here

for i in c:  # iterates over c
    for j in degree:  # iterates over degree
        for k in kernel:  # iterates kernel
            for df in decision_function_shape:  # iterates over decision_function_shape

                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=i, degree=j, kernel=k, decision_function_shape=df)

                # Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                # make the classifier prediction for each test sample and start computing its accuracy
                # --> add your Python code here
                accuracy = 0
                for db, row in enumerate(dbTest):
                    class_predicted = clf.predict([row[:-1]])[0]
                    if int(class_predicted) == int(row[-1]):
                        accuracy = accuracy + 1

                # check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                # Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                # --> add your Python code here

                accuracy = accuracy / len(dbTest)
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    highest_c = i
                    highest_degree = j
                    highest_kernel = k
                    highest_dfs = df
                    print("Highest SVM accuracy so far: ", highestAccuracy, ", Parameters: c=", highest_c, " degree=", highest_degree,
                          " kernel=", highest_kernel, " decision_function_shape= ", highest_dfs)

# print the final, highest accuracy found together with the SVM hyperparameters
# Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
# --> add your Python code here
print("Highest SVM accuracy: ", highestAccuracy,
      ", Parameters: c=", highest_c, " degree=", highest_degree, " kernel=", highest_kernel, " decision_function_shape = ", highest_dfs)
