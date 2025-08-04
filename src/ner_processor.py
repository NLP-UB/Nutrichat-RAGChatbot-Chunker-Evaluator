import spacy

class NERProcessor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def process_chunks(self, chunks):
        """
        Perform NER on each chunk and return modified text.
        Example: Append recognized entities to the chunk for better retrieval.
        """
        processed = []
        for chunk in chunks:
            doc = self.nlp(chunk)
            entities = [f"{ent.text}({ent.label_})" for ent in doc.ents]
            if entities:
                chunk = chunk + "\n\n[Entities: " + ", ".join(entities) + "]"
            processed.append(chunk)
        return processed
