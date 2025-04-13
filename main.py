from fastapi import FastAPI
from pydantic import BaseModel
import carpricepred

app = FastAPI()
model = carpricepred.LGB()

class CarInput(BaseModel):
    Year: int
    Present_price: float
    Kms_driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
