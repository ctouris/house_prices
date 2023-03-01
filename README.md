# house_prices
This code is a script for a machine learning project, specifically for predicting house prices using the Random Forest Regressor model.

The first part of the code reads in the training data from a CSV file, selects the target variable (SalePrice) and the features
(a subset of the available columns), and splits the data into training and validation sets.

Next, a Random Forest Regressor model is created, trained on the training set, and used to make predictions on the validation set. The mean
absolute error (MAE) between the predicted and actual values is calculated as a measure of model accuracy.

In the second part of the code, a new Random Forest Regressor model is created, trained on the full training data, and used to make predictions 
on a test set (stored in a separate CSV file). The predictions are saved to a new CSV file for submission.
