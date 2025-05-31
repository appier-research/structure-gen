import json
from pydantic import BaseModel
import pandas as pd
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser


class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)


class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)


class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt,
                                method='openai',
                                model_name='claude-3-haiku-20240307'
                            )

    def parse_answer(self, response, row):
        """
        Extract the following response final answer, only number with no symbol no comma or full stop, only the numeric value. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER NUMBER!
        Response: 
        <response>
        Answer:
        """
        if 'The answer is' in response:
            predict = response.split("The answer is")[-1]
            predict = predict.replace('.','').replace(' ','').replace('-','').lower().strip()
        else:
            predict = self.parser.parse(response)

        predict = str(predict)
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': predict
        }

class TwoStageLLM(BaseTextPrompter):
    """
    Run the following sglang service first
    $ docker run --gpus "device=0" \
        --shm-size 32g \
        -p 30003:30003 \
         -v /srv/home/ray-tam/hf_cache:/root/.cache/huggingface \
         --env "HF_TOKEN=XXX" \
        --ipc=host \
        lmsysorg/sglang:latest \
        python3 -m sglang.launch_server --model-path osmosis-ai/Osmosis-Structure-0.6B --host 0.0.0.0 --port 30003
    """
    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)
        from openai import OpenAI

        self.parser_prompt = self.config['parser_prompt']['text']
        api_key = "osmosis"
        api_base_url = "http://0.0.0.0:30003/v1"
        self.parser = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )

    def parse_answer(self, response, row):
        """
        Extract the following response final answer, only number with no symbol no comma or full stop, only the numeric value. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER NUMBER!
        Response: 
        <response>
        Answer:
        """
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    # "reasoning": {"type": "string"},
                    "answer": {"type": "string"}
                },
                # "required": ["reasoning", "answer"]
                "required": ["answer"]
            }
        )
        response = self.parser.chat.completions.create(
            model="Osmosis/Osmosis-Structure-0.6B",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {json_schema}"
                },
                {
                    "role": "user", 
                    "content": response,
                },
            ],
            temperature=0.6,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reasoning_extraction", "schema": json.loads(json_schema)},
            },
        )
        try:
            parsed_results = json.loads(response.choices[0].message.content)
        except json.decoder.JSONDecodeError as e:
            parsed_results = {
                'answer': 'decode_error'
            }
        predict = parsed_results['answer']
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': parsed_results
        }

class StructJSONPrompter(BaseJSONPrompter):
    schema = {
            "type": "function",
            "function": {
                "name": "get_reasoning_answer",
                "description": "Answer to the last question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "think step by step reasoning when solving question"
                        },
                        "final_answer": {
                            "type": "integer",
                            "description": "final answer number"
                        }
                    },
                    "required": ["reason", "final_answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            try:
                parsed_results = json.loads(parsed_results)
            except json.decoder.JSONDecodeError:
                parsed_results = {'final_answer': None }
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'final_answer': None }
        # we reassign it
        if 'answer' in parsed_results:
            parsed_results['final_answer'] = parsed_results['answer']
        if 'final_answer' not in parsed_results:
            parsed_results['final_answer'] = None
        # exact match with final_answer
        predict = parsed_results['final_answer']
        if isinstance(predict, str):
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}

        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': parsed_results,
            'parse_failed': parse_failed,
            'response_non_json': response_non_json,
            **additional_fields
        }

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'

            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer:\n```json\n{}\n```\n".format(
                    example['question'], json.dumps(example['response'], indent=4)
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question,
            'tools': [self.schema]
        }

        return self.template.render(data), data

class GSM8KStructureV1(BaseModel):
    reason: str
    answer: int

class GSM8KStructureV2(BaseModel):
    step_by_step_reasoning: str
    answer: int

class GSM8KStructureV3(BaseModel):
    think_step_by_step: str
    answer: int

class Step(BaseModel):
    explanation: str
    output: str

class GSM8KStructureV4(BaseModel):
    steps: list[Step]
    answer: int

class OAIStructPrompter(BaseJSONPrompter):

    def __init__(self, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.schema = GSM8KStructureV1
        if 'f1' in template_src:
            self.schema = GSM8KStructureV1
        if 'f2' in template_src:
            self.schema = GSM8KStructureV2
        if 'f3' in template_src:
            self.schema = GSM8KStructureV3
        if 'f4' in template_src:
            self.schema = GSM8KStructureV4

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            try:
                parsed_results = json.loads(parsed_results)
            except json.decoder.JSONDecodeError:
                parsed_results = {'final_answer': None }
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'final_answer': None }
        # we reassign it
        if 'answer' in parsed_results:
            parsed_results['final_answer'] = parsed_results['answer']
        if 'final_answer' not in parsed_results:
            parsed_results['final_answer'] = None
        # exact match with final_answer
        predict = parsed_results['final_answer']
        if isinstance(predict, str):
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}

        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': parsed_results,
            'parse_failed': parse_failed,
            'response_non_json': response_non_json,
            **additional_fields
        }

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'

            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer:\n```json\n{}\n```\n".format(
                    example['question'], json.dumps(example['response'], indent=4)
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question,
            'tools': self.schema
        }
        return self.template.render(data), data


class TwoStagePrompter(BaseJSONPrompter):

    def __init__(self, struct_format, num_shots=8, template_src='tasks/templates/gsm8k.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.format_instruct = self.config['format_instruct'][struct_format]

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            parsed_results = json.loads(parsed_results)
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'final_answer': None }
        if 'final_answer' not in parsed_results:
            parsed_results['final_answer'] = None
        # exact match with final_answer
        predict = parsed_results['final_answer']
        if isinstance(predict, str):
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}

        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': parsed_results,
            'parse_failed': parse_failed,
            'response_non_json': response_non_json,
            **additional_fields
        }

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'

            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer:\n```json\n{}\n```\n".format(
                    example['question'], json.dumps(example['response'], indent=4)
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': "Look carefully at the latest question and answer according to the task specification",
            'question': 'Question: '+question
        }

        return self.template.render(data), data


def parse_csv(response, row):
    parse_failed = 0
    response_non_csv = 0

    try:
        if '```' in response:
            predict = response.replace('```csv','```').split('```')[1]
        else:
            predict = response
    except IndexError:
        predict = 'key,value\nreasoning,""\nanswer,""'
        response_non_csv += 1

    if 'reasoning: 'not in response:
        predict = 'key,value\nreasoning,""\nanswer,""'
        response_non_csv += 1

    try:
        parsed_results = pd.read_csv(io.StringIO(predict))
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        parsed_results = pd.read_csv(io.StringIO('key,value\nreasoning,""\nanswer,""'))
        response_non_csv += 1

    if 'answer' not in parsed_results:
        predict = None
    else:
        predict = str(parsed_results['answer'])
        predict = predict.replace(',','').replace('$','').split(' ')[0]
    predict = str(predict).lower()
    answer = row['answer'].split('####')[-1].strip()
    correct = predict == answer
    return {
        'correct': correct,
        'answer': answer,
        'predict': predict,
        'parsed_result': dict(parsed_results),
        'parse_failed': parse_failed,
        'response_non_csv': response_non_csv
    }


def text_cot(question, num_shots=8):

    
    for example in FEWSHOTS[:num_shots]:
        prompt += "Q: {}\nA: {}. The answer is {}.\n".format(
            example['question'], example['response']['reason'], example['response']['answer']
        )
    prompt += "Q: "+question+"\nA:"
    return prompt, parse_text

def text_cot_2(question, num_shots=8):
    prompt = 'Answer the last question only\n'
    for example in FEWSHOTS[:num_shots]:
        prompt += "Q: {}\nA: {}. The answer is {}.\n".format(
            example['question'], example['response']['reason'], example['response']['answer']
        )
    prompt += "Q: "+question+"\nA:"
    return prompt, parse_text


def json_cot(question, num_shots=8):
    prompt = "output your result in json\n"
    for example in FEWSHOTS[:num_shots]:
        prompt += "Q: {}\nA:\n```json\n{}\n```\n".format(
            example['question'], json.dumps(example['response'], indent=4)
        )
    prompt += "Q: "+question+"\nA:\n"
    return prompt, parse_json

def yml_cot(question, num_shots=8):
    prompt = "output your result in yml\n"
    for example in FEWSHOTS[:num_shots]:
       prompt += 'Q: {}\nA:\n```yaml\nreasoning: {}\nanswer: "{}"\n```\n'.format(
           example['question'], example['response']['reason'], example['response']['answer']
       )
    prompt += "Q: "+question+"\nA:\n"
    return prompt, parse_yaml

def csv_cot(question, num_shots=8):
    prompt = "output your result in csv\n"
    for example in FEWSHOTS[:num_shots]:
       prompt += 'Q: {}\nA:\n```csv\nkey,value\nreasoning,"{}"\nanswer,{}\n```\n'.format(
           example['question'], example['response']['reason'], example['response']['answer']
       )
    prompt += "Q: "+question+"\nA:\n"
    return prompt, parse_csv

def json_hybrid_cot(question, num_shots=8):
    prompt = "output your result in reasoning answer first followed by a json with answer\n"
    for example in FEWSHOTS[:num_shots]:
        prompt += "Q: {}\nA: {}\n```json\n{}\n```\n".format(
            example['question'], example['response']['reason'], json.dumps({'answer': example['response']['answer']}, indent=4)
        )
    prompt += "Q: "+question+"\nA:"
    return prompt, parse_json