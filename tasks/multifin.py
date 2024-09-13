import json
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser

class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/multfin.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        result['correct'] = result['answer'] == result['parsed_result']['answer']
        result['predict'] = result['parsed_result']['answer']
        return result

class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/multfin.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        result['correct'] = result['answer'] == result['parsed_result']['root']['answer'][0]['_text']
        result['predict'] = result['parsed_result']['root']['answer'][0]['_text']
        return result

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer:\n```xml\n<root>\n  <answer>{}</answer>\n</root>\n```\n".format(
                    example['question'], example['response']['answer']
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question
        }

        return self.template.render(data)

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/multfin.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        if 'answer' not in result['parsed_result']:
            result['correct'] = False
            result['predict'] = None
        else:
            result['correct'] = result['answer'] == result['parsed_result']['answer']
            result['predict'] = result['parsed_result']['answer']
        return result

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                fewshot_text += 'Question: {}\nAnswer:\n```yaml\nanswer: "{}"\n```\n'.format(
                    example['question'], example['response']['answer']
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question
        }

        return self.template.render(data)

class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/multifin.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt,
                                method='claude',
                                model_name='claude-3-haiku-20240307'
                            )

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer: The answer is {}.\n".format(
                    example['question'], example['response']['answer']
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question
        }

        return self.template.render(data)

    def parse_answer(self, response, row):
        parser_rule = True
        if 'Answer:' in response:
            predict = response.split("Answer:")[-1]
            predict = predict.strip()
        elif 'The answer is' in response:
            predict = response.split("The answer is")[-1]
            predict = predict.strip()
        else:
            parser_rule = False
            predict = self.parser.parse(response)

        if parser_rule and len(predict) > 23:
            predict = self.parser.parse(predict)
            parser_rule = False

        predict = predict.replace('*','').replace('#','').strip().replace('.','').replace('<','').replace('>','')
        # access the parsed_result here to calculate jaccard metrics
        correct = row['answer'] == predict
        return {
            'correct': correct,
            'answer': row['answer'],
            'predict': predict,
            'parser_rule': parser_rule,
            'parsed_result': predict
        }

class StructJSONPrompter(BaseJSONPrompter):
    schema = {
            "type": "function",
            "function": {
                "name": "get_category",
                "description": "Answer to the last question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["Finance", "Technology", "Tax and Accounting", "Business and Management", "Government and Controls", "Industry"],
                            "description": "content most suited category"
                        },
                    },
                    "required": ["answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/multifin.yaml') -> None:
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

