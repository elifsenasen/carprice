from fastapi import FastAPI
from pydantic import BaseModel
import carpricepred
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Güvenlik için sonra sadece frontend IP'nizi yazın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarInput(BaseModel):
    Year: int
    Present_price: float
    Kms_driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int


IMG_DIR = "plots"
os.makedirs(IMG_DIR, exist_ok=True)

df=carpricepred.dataframe()
@app.get("/")
def home():
    return {"message": "Car Price Prediction Plot API"}

@app.get("/plot/km")
def plot_km():
    file_path = os.path.join(IMG_DIR, "km_plot.png")
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x='Kms_Driven', y='Selling_Price', data=df, hue="Fuel_Type", palette="coolwarm", edgecolor="black")
    plt.xticks(rotation=45)
    plt.xlabel("Kilometre (KM)")
    plt.ylabel("Selling Price")
    plt.title("Changing Selling Price According KM")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return FileResponse(file_path, media_type='image/png')

@app.get("/plot/age")
def plot_age():
    file_path = os.path.join(IMG_DIR, "age_plot.png")
    mean_price_age = df.groupby("age")["Selling_Price"].mean().reset_index()
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=mean_price_age['age'], y=mean_price_age['Selling_Price'], marker="o", color="b")
    plt.xlabel("Age")
    plt.ylabel("Mean Price")
    plt.title("Mean Selling Price According Age")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return FileResponse(file_path, media_type='image/png')

@app.get("/plot/fuel")
def plot_fuel():
    file_path = os.path.join(IMG_DIR, "fuel_plot.png")
    mean_fuel_price = df.groupby("Fuel_Type")["Selling_Price"].mean().reset_index()
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 4))
    sns.barplot(x=mean_fuel_price['Fuel_Type'], y=mean_fuel_price['Selling_Price'])
    plt.xlabel("Fuel")
    plt.ylabel("Mean Price")
    plt.title("Mean Selling Price According Fuel")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    return FileResponse(file_path, media_type='image/png')

@app.get("/plot/outliers")
def outlierplot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y=df["Selling_Price"], ax=axes[0])
    axes[0].set_title("Selling Price Outliers")

    sns.boxplot(y=df["Kms_Driven"], ax=axes[1])
    axes[1].set_title("KM Driven Outliers")

    sns.boxplot(y=df["age"], ax=axes[2])
    axes[2].set_title("Age Outliers")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/plot/removed_outliers")
def outlierplot_removed():
    new_df=carpricepred.remove_outlier(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(y=new_df["Selling_Price"], ax=axes[0])
    axes[0].set_title("Selling Price Outliers")

    sns.boxplot(y=new_df["Kms_Driven"], ax=axes[1])
    axes[1].set_title("KM Driven Outliers")

    sns.boxplot(y=new_df["age"], ax=axes[2])
    axes[2].set_title("Age Outliers")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict/{model_name}")
async def predict(model_name:str, car: CarInput):
    if model_name == "lgb":
        selected_model = carpricepred.LGB()
    elif model_name == "rf":
        selected_model = carpricepred.random_forest()
    elif model_name == "lr":
        selected_model = carpricepred.linearRegression()
    elif model_name == "xgb":
        selected_model = carpricepred.xgBoost()
    elif model_name == "svr":
        selected_model = carpricepred.svr()
    else:
        return {"error": "Model not found. Choose from: lgb, rf, lr, xgb, svr"}
    predicted_price=carpricepred.newinput(selected_model,car)
    return {"predicted_price": float(predicted_price)}

@app.post("/evaluate/{model_name}")
async def evaluate(model_name:str):
    if model_name == "lgb":
        selected_model = carpricepred.LGB()
    elif model_name == "rf":
        selected_model = carpricepred.random_forest()
    elif model_name == "lr":
        selected_model = carpricepred.linearRegression()
    elif model_name == "xgb":
        selected_model = carpricepred.xgBoost()
    elif model_name == "svr":
        selected_model = carpricepred.svr()
    else:
        return {"error": "Model not found. Choose from: lgb, rf, lr, xgb, svr"}
    results=carpricepred.evulatemodel(selected_model)
    return{"results": results}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)