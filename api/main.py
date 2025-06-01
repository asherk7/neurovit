from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, chat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(chat.router)
