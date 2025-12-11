import re
import ollama

class Generator:
    def __init__(self, model_name="gpt-oss", format="1 paragraf ringkas padat jelas"):
        self.model_name = model_name
        self.format = format

    def generate(self, context, question):
        formatted_prompt = f"Pertanyaan: {question}\n\nKonteks: {context} \n\nFormat jawaban: {format}"

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_content = response["message"]["content"]

        # hapus isi <think>...</think>
        final_answer = re.sub(
            r"<think>.*?</think>", "", response_content, flags=re.DOTALL
        ).strip()

        return final_answer