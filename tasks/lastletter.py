import json
from pydantic import BaseModel
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser

class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/lastletter.yaml') -> None:
        super().__init__(template_src, num_shots)


class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/lastletter.yaml') -> None:
        super().__init__(template_src, num_shots)

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/lastletter.yaml') -> None:
        super().__init__(template_src, num_shots)


class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/lastletter.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt,
                                method='claude',
                                model_name='claude-3-haiku-20240307'
                            )

    def parse_answer(self, response, row):
        """
        Extract the following response final answer, only lower cased letter with no space no comma or full stop, only the alphabet value a-z. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER!
        Response: 
        <response>
        Answer:
        """
        if 'The answer is' in response:
            predict = response.split("The answer is")[-1]
            predict = predict.replace('.','').replace('-','').lower().strip()
        elif 'The final answer is' in response:
            predict = response.split("The final answer is")[-1]
            predict = predict.replace('.','').replace('-','').lower().strip()
        else:
            predict = self.parser.parse(response)

        if '\n' in predict:
            predict = predict.split('\n')[0]
        predict = predict.replace('<answer>','').replace('</answer>','')
        predict = str(predict).replace(' ','').replace('*','').replace(':','').replace('"','')
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
                "name": "get_reasoning_answer",
                "description": "Answer to the last question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "think step by step when going through last letter of each words"
                        },
                        "final_answer": {
                            "type": "string",
                            "description": "final answer in lower case alphabet"
                        }
                    },
                    "required": ["reason", "final_answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/letter.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        try:
            if isinstance(parsed_results, str):
                parsed_results = json.loads(parsed_results)
        except json.decoder.JSONDecodeError:
            parse_failed += 1
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'final_answer': None }
        if 'answer' in parsed_results:
            parsed_results['final_answer'] = parsed_results['answer']
        if 'final_answer' not in parsed_results:
            parsed_results['final_answer'] = None
        # exact match with final_answer
        predict = parsed_results['final_answer']
        predict = str(predict).lower()
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

class LastLetterStructureV1(BaseModel):
    reason: str
    answer: str

class LastLetterStructureV2(BaseModel):
    step_by_step_reasoning: str
    answer: str

class LastLetterStructureV3(BaseModel):
    think_step_by_step: str
    answer: str

class Step(BaseModel):
    explanation: str
    output: str

class LastLetterStructureV4(BaseModel):
    steps: list[Step]
    answer: str


class OAIStructPrompter(BaseJSONPrompter):

    def __init__(self, num_shots=8, template_src='tasks/templates/letter.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.schema = LastLetterStructureV1
        if 'f1' in template_src:
            self.schema = LastLetterStructureV1
        elif 'f2' in template_src:
            self.schema = LastLetterStructureV2
        elif 'f3' in template_src:
            self.schema = LastLetterStructureV3
        elif 'f4' in template_src:
            self.schema = LastLetterStructureV4

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        try:
            if isinstance(parsed_results, str):
                parsed_results = json.loads(parsed_results)
        except json.decoder.JSONDecodeError:
            parse_failed += 1
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'final_answer': None }
        if 'answer' in parsed_results:
            parsed_results['final_answer'] = parsed_results['answer']
        if 'final_answer' not in parsed_results:
            parsed_results['final_answer'] = None
        # exact match with final_answer
        predict = parsed_results['final_answer']
        predict = str(predict).lower()
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
