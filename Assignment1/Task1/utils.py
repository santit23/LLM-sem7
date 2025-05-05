import spacy
from nltk.stem import PorterStemmer

# Load spacy model
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

def lemmatizer(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def POS_tagger(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]