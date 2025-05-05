from fastapi import FastAPI
from pydantic import BaseModel
import utils

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    operation: str

@app.post("/process_text")
async def process_text(text_request: TextRequest):
    text = text_request.text
    operation = text_request.operation

    if operation == "tokenize":
        result = utils.tokenizer(text)
    elif operation == "lemmatize":
        result = utils.lemmatizer(text)
    elif operation == "pos_tag":
        result = utils.POS_tagger(text)
    elif operation == "ner":
        result = utils.ner(text)
    else:
        return {"error": "Invalid operation"}

    return {"result": result}