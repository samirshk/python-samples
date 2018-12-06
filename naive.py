from sklearn.naive_bayes import GaussianNB

import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])

print(x)

Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

print(Y)

#Create a Gaussian Classifier
model = GaussianNB()

print(model)

# Train the model using Training sets

model.fit(x, Y)

print(model.predict([[-3,4], [6,8], [-20, 20]]))

print(model)