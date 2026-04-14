from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="My Credit Scoring API")

# Path to your actual model file
MODEL_PATH = "models/random_forest.joblib"
model = joblib.load(MODEL_PATH)

# Use SPACES here to match the training data columns
class ApplicantData(BaseModel):
    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str = pd.NA # Matches 'Saving accounts'
    Checking_account: str = pd.NA # Matches 'Checking account'
    Credit_amount: int # Matches 'Credit amount'
    Duration: int
    Purpose: str

@app.get("/")
def home():
    return {"message": "Credit Scoring API is Online", "status": "Ready"}

@app.post("/predict")
def predict(data: ApplicantData):
    # Create a dictionary and MANUALLY fix the keys to include spaces
    raw_data = data.dict()
    formatted_data = {
        "Age": raw_data["Age"],
        "Sex": raw_data["Sex"],
        "Job": raw_data["Job"],
        "Housing": raw_data["Housing"],
        "Saving accounts": raw_data["Saving_accounts"], # Added space
        "Checking account": raw_data["Checking_account"], # Added space
        "Credit amount": raw_data["Credit_amount"], # Added space
        "Duration": raw_data["Duration"],
        "Purpose": raw_data["Purpose"]
    }
    
    input_df = pd.DataFrame([formatted_data])
    
    # Run the prediction
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "risk_probability": round(float(probability), 4),
        "decision": "Reject" if probability > 0.5 else "Approve"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)