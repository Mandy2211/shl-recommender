from fastapi import FastAPI

from app.routes.chat import router


app = FastAPI()

app.include_router(router)


@app.get("/health")
def health():

    return {
        "status": "ok"
    }