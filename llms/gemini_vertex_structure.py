import os
import logging
import vertexai
import json
from time import sleep
from vertexai.preview.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from .utils import retry_with_exponential_backoff
vertexai.init(project=os.environ['GCP_PROJECT_NAME'], location="us-central1")

class GeminiStructure():
    SAFETY_SETTINGS={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }


    def __init__(self, model_name='gemini-1.0-pro-vision-001') -> None:
        self.model = GenerativeModel(model_name)

    @retry_with_exponential_backoff
    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, top_p=1, top_k=1) -> str:
        
        schema = schemas[0]
        fields = {}
        description = ''
        for key, metadata in schema['function']['parameters']['properties'].items():
            fields[key] = metadata['type']
            description += '{} - {}'.format(key, metadata['description'])
        schema_str = json.dumps(fields)
        postfix = f"Using this JSON schema:\n  Answer = {schema_str}\nReturn a `Answer`\nDescription:\n{description}\n"
        
        result = self.model.generate_content(
                prompt+'\n'+postfix,
                generation_config={
                    "max_output_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "response_mime_type": "application/json",
                    "top_k": int(top_k)
                },
                safety_settings=self.SAFETY_SETTINGS,
                stream=False
        ).candidates[0].content.parts[0].text
        res_info = {
            "input": prompt+'\n'+postfix,
            "output": result,
            "num_input_tokens": self.model.count_tokens(prompt).total_tokens,
            "num_output_tokens": self.model.count_tokens(result).total_tokens,
            "logprobs": []  # NOTE: currently the Gemini API does not provide logprobs
        }
        result = json.loads(result)
        return result, res_info

if __name__ == "__main__":
    llm = GeminiStructure('gemini-1.5-flash')
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
Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"""
    res_text, res_info = llm(prompt=prompt, schemas=[schema])
    print(res_text)
    print()
    from pprint import pprint
    pprint(res_info)
