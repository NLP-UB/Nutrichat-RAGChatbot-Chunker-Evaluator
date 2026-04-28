import re
import ollama

class Generator:
    def __init__(self, model_name="gpt-oss"):
        self.model_name = model_name

    def generate(self, context, question):
        formatted_prompt = f"Pertanyaan: {question}\n\nKonteks: {context} \n\nFormat jawaban: 1 paragraf ringkas padat jelas"

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_content = response["message"]["content"]

        final_answer = re.sub(
            r"<think>.*?</think>", "", response_content, flags=re.DOTALL
        ).strip()

        return final_answer