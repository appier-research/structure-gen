import os
from openai import OpenAI
import json
from .utils import retry_with_exponential_backoff

class OpenAIChat():
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
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            tool_choice="auto" if self.custom_model else "required",
            tools=schemas,
            **kwargs
        )
        res_text = response.choices[0].message.tool_calls[0].function.arguments
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens
        }
        return res_text, res_info

class OpenAIJSON():

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
        schema = schemas[0]
        fields = {}
        description = ''
        for key, metadata in schema['function']['parameters']['properties'].items():
            fields[key] = metadata['type']
            description += '{} - {}'.format(key, metadata['description'])
        schema_str = json.dumps(fields)
        postfix = f"Using this JSON schema:\n  Answer = {schema_str}\nReturn a `Answer`\nDescription:\n{description}\n"

        response = self.client.chat.completions.create(
            model=self.model_name,
            response_format={ "type": "json_object", "value": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."+'\n'+postfix},
                {'role': 'user', 'content': prompt}
            ],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
        )
        res_text = json.loads(response.choices[0].message.content)
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens
        }
        return res_text, res_info

if __name__ == "__main__":
    import json
    from typing import Any, Dict, List
    from pydantic import BaseModel, Field, create_model
    from enum import Enum, IntEnum
    # class EntityEnum(str, Enum):
    #     person = 'PER'
    #     location = 'LOC'
    #     misc = 'MISC'
    #     org = 'ORG'

    # class Entity(BaseModel):
    #     entity: EntityEnum = Field(description="entity type")
    #     text: str = Field(description="word in sentence")

    # class Output(BaseModel):
    #     reasoning: str = Field(description="step by step reasoning")
    #     entities: List[Entity]
    # print(json.dumps(Output.model_json_schema()))

    llm = OpenAIChat('gpt-3.5-turbo-0125')
    schema = {
            "type": "function",
            "function": {
                "name": "get_entity_from_sentence",
                "description": "Convert the question into the parameters value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "reasoning"
                        },
                        "entities": {
                            "type": "list[Entity]",
                            "description": 'List of entity enum, each Entity should be a dictionary {"type": entity, "text": word }'
                        }
                    },
                    "required": ["reason", "entities"]
                }
            }
        }

    prompt = """You are a NER converter who extract the named entity from the given sentence after format Question: <sentence>.
    Valid entities are : PERSON, ORGANIZATION, MISC, LOCATION
  Read the last question carefully and think step by step before answering, the final answer : entites must list of Entity schema.
  Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
  """
    # prompt = """You are a math tutor who helps students of all levels understand and solve mathematical problems.
    # Read the last question carefully and think step by step before answering, the final answer must be only a number.
    # Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    # You must output in tool
    # """
    # Output.model_json_schema()
    res_text, res_info = llm(prompt=prompt, schemas=[schema])
    print(res_text)
    print()
    from pprint import pprint
    pprint(res_info)
