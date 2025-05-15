import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
from pydantic import BaseModel


def dataframe():
    df= pd.read_csv("car data.csv")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        df = df.fillna(df.median())
    df['age'] = 2025 - df['Year']
    df.drop('Year',axis=1,inplace = True)
    return df

def remove_outlier(df):
    for col in ["Kms_Driven", "age", "Selling_Price"]:
        first_qntl=df[col].quantile(0.25)
        third_qntl=df[col].quantile(0.75)
        iqr=third_qntl-first_qntl
        lower_limit = first_qntl - 1.5 * iqr
        upper_limit=third_qntl+1.5*iqr
        df[col] = df[col].clip(lower=lower_limit, upper=df[col].quantile(0.90))
    return df

def final_df():
    df = dataframe()
    new_df = remove_outlier(df)
    return new_df

df=final_df()
scaler = MinMaxScaler(feature_range=(0, 1))
df[['Kms_Driven','age',"Present_Price"]] = scaler.fit_transform(df[['Kms_Driven','age',"Present_Price"]])
mean = df[['Kms_Driven', 'age', 'Present_Price']].mean()
std = df[['Kms_Driven', 'age', 'Present_Price']].std()

encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_feature_names)
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# set split
x=df.drop(columns=["Selling_Price","Car_Name"])
y=df["Selling_Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

def random_forest():
    rf = RandomForestRegressor()
    param_grid = {
        "n_estimators": list(range(400, 1000, 100)),
        "max_depth": list(range(5, 15, 2)),
        "min_samples_split": [4, 6, 8],
        "min_samples_leaf": [1,2,5,7],
        "max_features": ["log2", "sqrt", None]
    }
    rf_rs = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=30, cv=10, verbose=2, random_state=42)
    rf_rs.fit(x_train, y_train)

    # Best Parameters
    print("Best Parameters for:", rf_rs.best_params_)
    return rf_rs.best_estimator_

def linearRegression():
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    return lr

def xgBoost():
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(x_train,y_train)
    return xgb_model

def svr():
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(x_train, y_train)
    return svr

def LGB():
    lgb_model=lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1,min_split_gain=0.1)
    lgb_model.fit(x_train,y_train)
    return lgb_model

def evulatemodel(model):
    y_train_pred = model.predict(x_train)
    mse = mean_squared_error(y_train, y_train_pred)
    mae = mean_absolute_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    y_test_pred = model.predict(x_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return {
        "train_mse": mse, "train_mae": mae, "train_r2": r2,
        "test_mse": mse_test, "test_mae": mae_test, "test_r2": r2_test
    }


class Car(BaseModel):
    Year: int
    Present_price: float
    Kms_driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int

def newinput(model, car:Car):
    new_data = {
        'Year': car.Year,
        'Present_Price': car.Present_price,
        'Kms_Driven': car.Kms_driven,
        'Fuel_Type': car.Fuel_Type,
        'Seller_Type': car.Seller_Type,
        'Transmission': car.Transmission,
        'Owner': car.Owner
    }

    new_df = pd.DataFrame([new_data])
    new_df['age'] = 2025 - new_df['Year']
    new_df.drop('Year', axis=1, inplace=True)

    encoded_cols = encoder.transform(new_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_df], axis=1)

    new_df = new_df.reindex(columns=x_train.columns, fill_value=0)

    new_df[['Kms_Driven', 'age', 'Present_Price']] = scaler.transform(new_df[['Kms_Driven', 'age', 'Present_Price']])

    prediction = model.predict(new_df)
    return prediction[0]

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import lightgbm as lgb
from pydantic import BaseModel


def dataframe():
    df= pd.read_csv("car data.csv")
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        df = df.fillna(df.median())
    df['age'] = 2025 - df['Year']
    df.drop('Year',axis=1,inplace = True)
    return df

def remove_outlier(df):
    for col in ["Kms_Driven", "age", "Selling_Price"]:
        first_qntl=df[col].quantile(0.25)
        third_qntl=df[col].quantile(0.75)
        iqr=third_qntl-first_qntl
        lower_limit = first_qntl - 1.5 * iqr
        upper_limit=third_qntl+1.5*iqr
        df[col] = df[col].clip(lower=lower_limit, upper=df[col].quantile(0.90))
    return df

def final_df():
    df = dataframe()
    new_df = remove_outlier(df)
    return new_df

df=final_df()
scaler = MinMaxScaler(feature_range=(0, 1))
df[['Kms_Driven','age',"Present_Price"]] = scaler.fit_transform(df[['Kms_Driven','age',"Present_Price"]])
mean = df[['Kms_Driven', 'age', 'Present_Price']].mean()
std = df[['Kms_Driven', 'age', 'Present_Price']].std()

encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_feature_names)
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# set split
x=df.drop(columns=["Selling_Price","Car_Name"])
y=df["Selling_Price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

def random_forest():
    rf = RandomForestRegressor()
    param_grid = {
        "n_estimators": list(range(400, 1000, 100)),
        "max_depth": list(range(5, 15, 2)),
        "min_samples_split": [4, 6, 8],
        "min_samples_leaf": [1,2,5,7],
        "max_features": ["log2", "sqrt", None]
    }
    rf_rs = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=30, cv=10, verbose=2, random_state=42)
    rf_rs.fit(x_train, y_train)

    # Best Parameters
    print("Best Parameters for:", rf_rs.best_params_)
    return rf_rs.best_estimator_

def linearRegression():
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    return lr

def xgBoost():
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(x_train,y_train)
    return xgb_model

def svr():
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(x_train, y_train)
    return svr

def LGB():
    lgb_model=lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1,min_split_gain=0.1)
    lgb_model.fit(x_train,y_train)
    return lgb_model

def evulatemodel(model):
    y_train_pred = model.predict(x_train)
    mse = mean_squared_error(y_train, y_train_pred)
    mae = mean_absolute_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    y_test_pred = model.predict(x_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return {
        r2_test
    }


class Car(BaseModel):
    Year: int
    Present_price: float
    Kms_driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int

def newinput(model, car:Car):
    new_data = {
        'Year': car.Year,
        'Present_Price': car.Present_price,
        'Kms_Driven': car.Kms_driven,
        'Fuel_Type': car.Fuel_Type,
        'Seller_Type': car.Seller_Type,
        'Transmission': car.Transmission,
        'Owner': car.Owner
    }

    new_df = pd.DataFrame([new_data])
    new_df['age'] = 2025 - new_df['Year']
    new_df.drop('Year', axis=1, inplace=True)

    encoded_cols = encoder.transform(new_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_df], axis=1)

    new_df = new_df.reindex(columns=x_train.columns, fill_value=0)

    new_df[['Kms_Driven', 'age', 'Present_Price']] = scaler.transform(new_df[['Kms_Driven', 'age', 'Present_Price']])

    prediction = model.predict(new_df)
    return prediction[0]

