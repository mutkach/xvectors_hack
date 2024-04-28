from typing import Union
from fastapi import FastAPI

from tritonclient.http import InferenceServerClient
import redis as rd

app = FastAPI()
client = rd.Redis(host="localhost", port=6379, decode_responses=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/lookup/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}