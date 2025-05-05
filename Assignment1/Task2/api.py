from fastapi import FastAPI
from pydantic import BaseModel
from tfidf_utils import get_embedding

app = FastAPI()

class WordRequest(BaseModel):
    word: str

@app.post("/embedding")
def embedding_api(req: WordRequest):
    embedding = get_embedding
    if embedding is not None:
        return {"word": req.word, "embedding": embedding.tolist()}
    return {"error": "Word not found in vocabulary"}
