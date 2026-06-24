from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum


class SexEnum(str, Enum):
    male = "male"
    female = "female"


class HousingEnum(str, Enum):
    own = "own"
    free = "free"
    rent = "rent"


class SavingAccountsEnum(str, Enum):
    little = "little"
    moderate = "moderate"
    quite_rich = "quite rich"
    rich = "rich"
    unknown = "unknown"


class CheckingAccountEnum(str, Enum):
    little = "little"
    moderate = "moderate"
    rich = "rich"
    unknown = "unknown"


class PurposeEnum(str, Enum):
    car = "car"
    furniture_equipment = "furniture/equipment"
    radio_tv = "radio/TV"
    domestic_appliances = "domestic appliances"
    repairs = "repairs"
    education = "education"
    business = "business"
    vacation_others = "vacation/others"


class CreditRiskFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Yaş", example=30)
    sex: SexEnum = Field(..., description="Cinsiyet", example="male")
    job: int = Field(..., ge=0, le=3, description="Meslek kodu (0=unskilled & non-resident, 1=unskilled & resident, 2=skilled, 3=highly skilled)", example=2)
    housing: HousingEnum = Field(..., description="Konut durumu", example="own")
    saving_accounts: SavingAccountsEnum = Field(..., description="Tasarruf hesabı durumu", example="little")
    checking_account: CheckingAccountEnum = Field(..., description="Vadesiz hesap durumu", example="moderate")
    credit_amount: int = Field(..., gt=0, description="Kredi tutarı (EUR)", example=5000)
    duration: int = Field(..., gt=0, description="Kredi vadesi (ay)", example=24)
    purpose: PurposeEnum = Field(..., description="Kredi amacı", example="car")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 30,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        }
    }


class PredictionInput(BaseModel):
    features: List[CreditRiskFeatures]


class PredictionOutput(BaseModel):
    predictions: List[int] = Field(..., description="Tahmin sonuçları: 0=bad risk, 1=good risk")


class ProbabilityOutput(BaseModel):
    probabilities: List[List[float]] = Field(
        ...,
        description="Her kayıt için [P(bad), P(good)] formatında olasılıklar"
    )
