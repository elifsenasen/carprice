import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression



df= pd.read_csv("car data.csv")

missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    df = df.fillna(df.median())
    
df['age'] = 2025 - df['Year']
df.drop('Year',axis=1,inplace = True)
#print(df)

#km plot
sns.set_style("darkgrid")
plt.figure(figsize=(10, 4))

# x axis km_driven
sns.scatterplot(x='Kms_Driven', y='Selling_Price', data=df, hue="Fuel_Type", palette="coolwarm", edgecolor="black")
plt.xticks(rotation=45)
plt.xlabel("Kilometre (KM)")
plt.ylabel("Selling Price")
plt.title("Changing Selling Price According KM")
plt.show()

#age plot
mean_price_age=df.groupby("age")["Selling_Price"].mean().reset_index()
sns.set_style("darkgrid")
plt.figure(figsize=(10, 4))
sns.lineplot(x=mean_price_age['age'], y=mean_price_age['Selling_Price'], marker="o", color="b")
plt.xlabel("Age")
plt.ylabel("Mean Price")
plt.title("Mean Selling Price According Age")
plt.show()

mean_fuel_price = df.groupby("Fuel_Type")["Selling_Price"].mean().reset_index()
sns.set_style("darkgrid")
plt.figure(figsize=(10, 4))
sns.barplot(x=mean_fuel_price['Fuel_Type'], y=mean_fuel_price['Selling_Price'])
plt.xlabel("Fuel")
plt.ylabel("Mean Price")
plt.title("Mean Selling Price According Fuel")
plt.show()

def outlierplot():
    plt.figure(figsize=(12, 6))
    # Selling Price Outliers
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df["Selling_Price"])
    plt.title("Selling Price Outliers")

    # KM Driven Outliers
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df["Kms_Driven"])
    plt.title("KM Driven Outliers")

    # Age Outliers
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df["age"])
    plt.title("Age Outliers")

    plt.tight_layout()
    plt.show()

def remove_outlier(columns):
    first_qntl=columns.quantile(0.25)
    third_qntl=columns.quantile(0.75)
    iqr=third_qntl-first_qntl
    lower_limit = first_qntl - 1.5 * iqr
    upper_limit=third_qntl+1.5*iqr
    columns = columns.clip(lower=lower_limit, upper=columns.quantile(0.90))
    return columns
    
outlierplot()
df["Kms_Driven"] = remove_outlier(df["Kms_Driven"])
df["age"] = remove_outlier(df["age"])
df["Selling_Price"] = remove_outlier(df["Selling_Price"])
outlierplot()

scaler = MinMaxScaler(feature_range=(0, 1))
df[['Kms_Driven', 'age',"Selling_Price","Present_Price"]] = scaler.fit_transform(df[['Kms_Driven','age',"Selling_Price","Present_Price"]])

le = LabelEncoder()

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
        
def evulatemodel(model):
    y_train_pred_best = model.predict(x_train)
    mse_best = mean_squared_error(y_train, y_train_pred_best)
    mae_best = mean_absolute_error(y_train, y_train_pred_best)
    r2_best = r2_score(y_train, y_train_pred_best)
    
    print("Train Model")
    print("Best Model - Mean Squared Error (MSE):", mse_best)
    print("Best Model - Mean Absolute Error (MAE):", mae_best)
    print("Best Model - R2 Score:", r2_best)
    
    y_test_pred_best=model.predict(x_test)
    mse_best = mean_squared_error(y_test, y_test_pred_best)
    mae_best = mean_absolute_error(y_test, y_test_pred_best)
    r2_best = r2_score(y_test, y_test_pred_best)
    
    print("Test Model")
    print("Best Model - Mean Squared Error (MSE):", mse_best)
    print("Best Model - Mean Absolute Error (MAE):", mae_best)
    print("Best Model - R2 Score:", r2_best)
    
    
def xgBoost():
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb_model.fit(x_train,y_train)
    return xgb_model
    
#lr_model = linearRegression()
#rf_model=random_forest()
xgb_model=xgBoost()
evulatemodel(xgb_model)