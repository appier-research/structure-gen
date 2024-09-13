import os
import xml
import json
import yaml
import pandas as pd
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser


class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/task280-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)

class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/task280-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/task280-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)

class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/task280-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt, method='claude', model_name='claude-3-haiku-20240307')

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                if 'reason' in example['response']:
                    fewshot_text += "Passage: {}\nAnswer: {} So the answer is {}.\n".format(
                        example['question'], example['response']['reason'], example['response']['answer']
                    )
                else:
                    fewshot_text += "Passage: {}\nAnswer: The answer is {}.\n".format(
                        example['question'], example['response']['answer']
                    )
        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': question
        }

        return self.template.render(data)

    def parse_answer(self, response, row):
        """
        Extract the following response final answer, only number with no symbol no comma or full stop, only the numeric value. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER NUMBER!
        Response: 
        <response>
        Answer:
        """
        parser_rule = False
        if 'So the answer is' in response:
            predict = response.split("So the answer is")[-1]
            predict = predict.strip()
            parser_rule = True
        else:
            predict = self.parser.parse(response)
        if parser_rule and len(predict) > 30:
            predict = self.parser.parse(predict)

        predict = str(predict).lower().replace('.','').replace('*','').replace('the answer is','').replace(':','').strip()
        answer = row['answer']
        correct = predict == answer
        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': predict
        }

class StructJSONPrompter(BaseJSONPrompter):
    schema = {
            "type": "function",
            "function": {
                "name": "get_category",
                "description": "Determine if the paragraph is one of these four category : gender, profession, race, and religion",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["gender", "profession", "race", "religion"],
                            "description": "paragraph most suited category"
                        },
                    },
                    "required": ["answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/task280.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        try:
            if isinstance(parsed_results, str):
                parsed_results = json.loads(parsed_results)
        except json.decoder.JSONDecodeError:
            parse_failed += 1
            parsed_results = {'answer': None }
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

