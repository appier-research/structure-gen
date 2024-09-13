import os
from openai import OpenAI
import json
from .utils import retry_with_exponential_backoff

class OpenAIStructureV2():
    TOP_LOGPROBS = 1

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        params = {'api_key': os.environ['OAI_KEY']}
        self.custom_model = False
        if os.getenv('CUSTOM_API_URL'):
            params['base_url'] = os.environ['CUSTOM_API_URL']
            self.custom_model = True
        self.client = OpenAI(**params)
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, top_p=0.999, **kwargs) -> tuple[str, dict]:
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "user", 'content': prompt}
            ],
            response_format=schemas,
        )
        event = response.choices[0].message.parsed.json()
        res_info = {
            "input": prompt,
            "output": event,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens
        }
        return event, res_info


if __name__ == "__main__":
    from pydantic import BaseModel
    class Response(BaseModel):
        reasoning: str
        answer: int

    llm = OpenAIStructureV2('gpt-4o-mini-2024-07-18')
    res, res_info = llm(prompt='Answer the following in the response format\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    schemas=Response)
    print(res)
