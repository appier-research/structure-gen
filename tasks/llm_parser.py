import os
from openai import OpenAI
from together import Together

class LLMParser():

    def __init__(self, parser_prompt, method='claude', model_name="meta-llama/Llama-3-8b-chat-hf"):
        if method == 'together' and 'TOGETHER_API_KEY' not in os.environ:
            method = 'local'
        self.model_name = model_name
        if method == 'local':
            assert 'TGI_URL' in os.environ
            self.client = OpenAI(base_url=os.environ['TGI_URL'], api_key="x")
        elif method == 'together':
            self.client = Together(api_key=os.environ['TOGETHER_API_KEY'])
        elif method == 'claude':
            from llms.claude import ClaudeChat
            self.client = ClaudeChat('claude-3-haiku-20240307')
            self.model_name = 'claude-3-haiku-20240307'
        elif method == 'openai':
            from llms.oai_chat import OpenAIChat
            self.client = OpenAIChat('gpt-4o-mini-2024-07-18')
            self.model_name = 'gpt-4o-mini-2024-07-18'
        else:
            raise ValueError('failed')
        self.method = method
        self.parser_prompt = parser_prompt

    def parse(self, response):
        if self.method in ('claude', 'openai'):
            text, res_info = self.client(self.parser_prompt+"\n"+response+"\nAnswer:")
            if 'error:' in text:
                raise ValueError()
            return text
        else:
            res = self.client.chat.completions.create(
                                    messages=[{
                                        "role": "user",
                                        "content": self.parser_prompt+"\n"+response+"\nAnswer:"
                                    }],
                                    model=self.model_name,
                                    temperature=0.0,
                                    max_tokens=100,
                                )
            return res.choices[0].message.content.strip()
