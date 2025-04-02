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
from sklearn.model_selection import GridSearchCV


df= pd.read_csv("CARS.csv")

missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    df = df.fillna(df.median())
    
df['age'] = 2025 - df['year']
df.drop('year',axis=1,inplace = True)
#print(df)

#km plot
# sns.set_style("darkgrid")
# plt.figure(figsize=(10, 4))
# # x axis km_driven
# sns.scatterplot(x='km_driven', y='selling_price', data=df, hue="fuel", palette="coolwarm", edgecolor="black")
# plt.xticks(rotation=45)
# plt.xlabel("Kilometre (KM)")
# plt.ylabel("Selling Price")
# plt.title("Changing Selling Price According KM")
# plt.show()

# #age plot
# mean_price_age=df.groupby("age")["selling_price"].mean().reset_index()
# sns.set_style("darkgrid")
# plt.figure(figsize=(10, 4))
# sns.lineplot(x=mean_price_age['age'], y=mean_price_age['selling_price'], marker="o", color="b")
# plt.xlabel("Age")
# plt.ylabel("Mean Price")
# plt.title("Mean Selling Price According Age")
# plt.show()

# mean_fuel_price = df.groupby("fuel")["selling_price"].mean().reset_index()
# sns.set_style("darkgrid")
# plt.figure(figsize=(10, 4))
# sns.barplot(x=mean_fuel_price['fuel'], y=mean_fuel_price['selling_price'])
# plt.xlabel("Fuel")
# plt.ylabel("Mean Price")
# plt.title("Mean Selling Price According Fuel")
# plt.show()

def outlierplot():
    plt.figure(figsize=(12, 6))
    # Selling Price Outliers
    plt.subplot(1, 3, 1)
    sns.boxplot(y=df["selling_price"])
    plt.title("Selling Price Outliers")

    # KM Driven Outliers
    plt.subplot(1, 3, 2)
    sns.boxplot(y=df["km_driven"])
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
    columns = columns.clip(lower=lower_limit, upper=columns.quantile(0.99))
    return columns
    
#outlierplot()
df["km_driven"] = remove_outlier(df["km_driven"])
df["age"] = remove_outlier(df["age"])
df["selling_price"] = remove_outlier(df["selling_price"])
#outlierplot()

scaler = MinMaxScaler(feature_range=(0, 1))
df[['km_driven', 'age',"selling_price"]] = scaler.fit_transform(df[['km_driven', 'age',"selling_price"]])

#random forest
le = LabelEncoder()
df['owner'] = df['owner'].replace({
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
    "Test Drive Car": 0
}).infer_objects(copy=False)

encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_cols = ['fuel', 'seller_type', 'transmission']
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_feature_names)
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

#print(df.head(60))
# print(df.columns)

x=df.drop(columns=["selling_price","name"])
y=df["selling_price"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
# model = RandomForestRegressor(random_state=42, max_depth=35)
# model.fit(x_train, y_train)
# y_pred=model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error (MSE):", mse)
# print("Mean Absolute Error (MAE):", mae)
# print("R2 Score:", r2)

rf = RandomForestRegressor()

param_grid = {
    "n_estimators": list(range(500, 1000, 100)),
    "max_depth": list(range(6, 12, 2)),  
    "min_samples_split": [4, 6, 8], 
    "min_samples_leaf": [1,2,5,7],
    "max_features": ["log2", "sqrt", None]
}

rf_rs = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, random_state=42)
rf_rs.fit(x_train, y_train)

# En iyi parametreler
print("Best Parameters:", rf_rs.best_params_)

# En iyi model ile tahmin yapma
best_model = rf_rs.best_estimator_
y_train_pred_best = best_model.predict(x_train)

# Değerlendirme
mse_best = mean_squared_error(y_train, y_train_pred_best)
mae_best = mean_absolute_error(y_train, y_train_pred_best)
r2_best = r2_score(y_train, y_train_pred_best)

print("Best Model - Mean Squared Error (MSE):", mse_best)
print("Best Model - Mean Absolute Error (MAE):", mae_best)
print("Best Model - R2 Score:", r2_best)