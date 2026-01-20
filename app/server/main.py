from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
from inference import ModelLoader

app = FastAPI()

# --- 1. CORS CONFIGURATION ---
origins = ["*"]  # Allow all for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_loader = ModelLoader()

# --- 2. UPDATE REQUEST/RESPONSE MODELS ---
class GenerateRequest(BaseModel):
    context: str
    slider_value: int = 50  # Added slider value
    max_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50

class GenerateResponse(BaseModel):
    bdh_text: str  # Response for Left Box
    gpt_text: str  # Response for Right Box

class ValidateRequest(BaseModel):
    context: str
    draft: str

class Contradiction(BaseModel):
    span: tuple[int, int]
    reason: str

class ValidateResponse(BaseModel):
    consistent: bool
    contradictions: Union[list[Contradiction], None] = None


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # 1. Generate Hybrid/BDH text using the slider
    bdh_result = model_loader.generate(
        prompt=request.context,
        slider_value=request.slider_value
    )
    
    # 2. Generate Baseline GPT-2 text (for comparison)
    gpt_result = model_loader.generate_text(
        context=request.context,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k
    )

    return {
        "bdh_text": bdh_result,
        "gpt_text": gpt_result
    }

@app.post("/api/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest):
    result = model_loader.validate(request.context, request.draft)
    return result

@app.get("/")
async def root():
    return {"status": "Project PlotArmor API is running"}