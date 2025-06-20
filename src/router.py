import spacy

nlp = spacy.load("en_core_web_sm")

def detect_use_graph(question: str) -> bool:
    doc = nlp(question)
    return any(ent.label_ in ("PERSON","ORG","GPE") for ent in doc.ents)
