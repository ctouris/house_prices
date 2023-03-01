import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# save filepath to variable
iowa_file_path = "D:\\Users\\Cecilia\\Documents\\DATA_ANALYST_SCIENCE_COURSES\\PROYECTS\\PROJECT_2\\train.csv"


# read the data and store data in DataFrame titled iowa_data
iowa_train_data = pd.read_csv(iowa_file_path)

# view data
# iowa_head = iowa_train_data.head()


# view columns
# col = iowa_train_data.columns


# Selecting The Prediction Target
y = iowa_train_data.SalePrice


# Choosing "Features"
iowa_features = ['MSSubClass', 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
                 'TotRmsAbvGrd', 'OverallQual', 'OverallCond']
X = iowa_train_data[iowa_features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)

print('\n')

print(list(rf_val_predictions[:5]))
print(list(val_y[:5]))
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print('\n')
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print('\n')

##########################################################


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_train = "D:\\Users\\Cecilia\\Documents\\DATA_ANALYST_SCIENCE_COURSES\\PROYECTS\\PROJECT_2\\train.csv"

# read the data and store data in DataFrame titled rf_model_on_full_train
rf_model_train = pd.read_csv(rf_model_on_full_train)

# Define a random forest model
rf_model_full = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data
rf_model_full.fit(X, y)

# Selecting The Prediction Target
y = rf_model_train.SalePrice

# Choosing "Features"
features_ = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr',
             'TotRmsAbvGrd']
X = rf_model_train[features_]

# Define a random forest model
rf_model_random = RandomForestRegressor()
rf_model_random.fit(X, y)

# path to file you will use for predictions
test_data_path = "D:\\Users\\Cecilia\\Documents\\DATA_ANALYST_SCIENCE_COURSES\\PROYECTS\\PROJECT_2\\test.csv"

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features

test_X = test_data[features_]

# make predictions which we will submit.
test_preds = rf_model_random.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

# print(output)
