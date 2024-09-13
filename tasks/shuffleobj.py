import os
import xml
import json
from enum import Enum
from pydantic import BaseModel, Field
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter

class StructJSONPrompter(BaseJSONPrompter):
    schema = {
            "type": "function",
            "function": {
                "name": "get_answer_choice",
                "description": "Answer to the last question",
                "parameters": {
                    "type": "object",
                    "properties": { # try a_reason if the llm would prfer to generate it first
                        "a_reason": {
                            "type": "string",
                            "description": "think step by step here"
                        },
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D", "E", "F", "G"],
                            "description": "content most suited category"
                        }
                    },
                    "required": ["a_reason", "answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/shuffleobj-t1-structure.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            parsed_results = json.loads(parsed_results)
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'answer': None }
        if 'answer' not in parsed_results:
            parsed_results['answer'] = None
        # exact match with answer
        predict = parsed_results['answer']
        answer = row['answer']
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

class Choice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class ShuffleObjStructureV1(BaseModel):
    reason: str
    answer: Choice = Field(..., description="most suited category")

class ShuffleObjStructureV2(BaseModel):
    step_by_step_reasoning: str
    answer: Choice = Field(..., description="most suited category")

class ShuffleObjStructureV3(BaseModel):
    think_step_by_step: str
    answer: Choice = Field(..., description="most suited category")

class Step(BaseModel):
    explanation: str
    output: str

class ShuffleObjStructureV4(BaseModel):
    steps: list[Step]
    answer: Choice = Field(..., description="most suited category")

class OAIStructPrompter(BaseJSONPrompter):

    def __init__(self, num_shots=8, template_src='tasks/templates/shuffleobj-t1-structure.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.schema = ShuffleObjStructureV1
        if 'f1' in template_src:
            self.schema = ShuffleObjStructureV1
        elif 'f2' in template_src:
            self.schema = ShuffleObjStructureV2
        elif 'f3' in template_src:
            self.schema = ShuffleObjStructureV3
        elif 'f4' in template_src:
            self.schema = ShuffleObjStructureV4

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            parsed_results = json.loads(parsed_results)
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'answer': None }
        if 'answer' not in parsed_results:
            parsed_results['answer'] = None
        # exact match with answer
        predict = parsed_results['answer']
        answer = row['answer']
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

