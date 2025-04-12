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

@app.post("/predict")
async def predict(car: CarInput):
    predicted_price=carpricepred.newinput(model,car)
    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)