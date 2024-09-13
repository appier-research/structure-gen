import os
import logging
from time import sleep
from anthropic import AnthropicVertex

class ClaudeChat():

    def __init__(self, model_name='claude-3-sonnet@20240229') -> None:
        self.client = AnthropicVertex(
                                region="asia-southeast1",
                                project_id=os.environ['GCP_PROJECT_NAME'],
                                access_token=os.environ['GCP_ACCESS_TOKEN']
                            )
        self.model_name = model_name

    def __call__(self, prompt, max_tokens=512, temperature=0.0, **kwargs) -> str:
        success = False
        failed = 0
        while not success:

            try:
                message = self.client.messages.create(
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    model=self.model_name,
                )
                result = message.content[0].text
                success = True
                sleep(5.0)
            except Exception as e:
                logging.error('anthropic:'+str(e))
                result = 'error:{}'.format(e)
                failed += 1
                sleep(5.0)
            if failed > 10:
                break
        return result


if __name__ == "__main__":
    client = ClaudeChat()
    print(client("Hi"))