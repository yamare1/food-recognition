from fastapi import FastAPI
from api.endpoints.food_routes import router as food_router

app = FastAPI(title="Food Recognition API")

app.include_router(food_router, prefix="/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Recognition API"}
