import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# print(data.head())

# creating a new dataframe with only elements that we want to use for our regression model
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# these are attributes
# we want to get labels based on attributes

# G3 will be our label to compute based on attributes
predict = "G3"

# setup two arrays, one for attributes, one for labels

x = np.array(data.drop([predict], 1))  # returns a new dataframe without G3
y = np.array(data[predict])  # only care about the G3 value


# this splits up our data into 4 different arrays
# we can't train the model off of our testing data, otherwise we will get inaccurate results (it will just be perfect)
# in this case the model would be able to perfectly estimate because it will have already seen that info
# splits up 10% of our data into test samples, so that when we test it will not have seen that info yet
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

