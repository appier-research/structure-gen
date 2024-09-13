import os
from openai import OpenAI

from .utils import retry_with_exponential_backoff

class OpenAIChat():
    TOP_LOGPROBS = 1

    def __init__(self, model_name='gpt-3.5-turbo-0125') -> None:
        params = {'api_key': os.environ['OAI_KEY']}
        if os.getenv('CUSTOM_API_URL') and 'gpt-' not in model_name:
            params['base_url'] = os.environ['CUSTOM_API_URL']
        self.client = OpenAI(**params)
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt, max_tokens=512, temperature=0.0, top_p=0.999, **kwargs) -> tuple[str, dict]:
        arguments = {
            'temperature': float(temperature),
            'max_tokens': int(max_tokens),
            'top_p': float(top_p),
            'logprobs': True,
            'top_logprobs': self.TOP_LOGPROBS,
        }
        if 'gpt' not in self.model_name:
            # O1 doesn't support these as of 09/12
            arguments.pop('max_tokens')
            arguments.pop('logprobs')
            arguments.pop('top_logprobs')
            arguments.pop('temperature')
            arguments.pop('top_p')
            # yeah might have just assign arguments={}

        kwargs = {**arguments, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs
        )
        if response.choices[0].logprobs is not None:
            log_prob_seq = response.choices[0].logprobs.content
            assert response.usage.completion_tokens == len(log_prob_seq)
        else:
            log_prob_seq = []
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq]
        }
        return res_text, res_info

if __name__ == "__main__":
    llm = OpenAIChat()
    res_text, res_info = llm(prompt="Say apple!")
    print(res_text)
    print()
    from pprint import pprint
    pprint(res_info)
