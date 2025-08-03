from transformers import pipeline

class Generator:
    def __init__(self, model_name='google/flan-t5-base'):
        self.model = pipeline('text2text-generation', model=model_name)

    def generate(self, context, question):
        prompt = f"Answer the question based on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
        result = self.model(prompt, max_length=256, truncation=True)
        return result[0]['generated_text']
