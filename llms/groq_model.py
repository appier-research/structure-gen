import os
from groq import Groq

from .utils import retry_with_exponential_backoff

class GroqModel:
    model_list = [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "llama2-70b-4096",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it"
    ]

    def __init__(self, model_name: str = "llama3-70b-8192") -> None:
        self.client = Groq(
            # This is the default and can be omitted
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, max_tokens=1024, temperature=0.0, **kwargs) -> tuple[str, dict]:
        res = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        res_text = res.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": res.usage.prompt_tokens,
            "num_output_tokens": res.usage.completion_tokens,
            "logprobs": []
        }
        return res_text, res_info

if __name__ == "__main__":
    llm = GroqModel(model_name="llama3-70b-8192")
    res_text, res_info = llm(prompt="Are you an instruction-tuned version of LLama-3?")
    print(res_text)
    print(res_info)
