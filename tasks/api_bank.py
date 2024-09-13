"""
Didn't finish in time
"""
import json
import yaml
from dict2xml import dict2xml
from .base import BaseJSONPrompter

def convert_desc2format(payload, format):
    if format == 'yaml':
        return [ yaml.dump(row, default_flow_style=False) for row in payload['available_functions'] ]
    elif format == 'json':
        return [ json.dumps(row, indent=4) for row in payload['available_functions'] ]
    elif format == 'xml':
        return [ dict2xml(row, wrap="root", indent="  ") for row in payload['available_functions'] ]
    else:
        return [ str(row) for row in payload['available_functions']]



class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/api_bank.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.method = 'json'

    def prompt(self, row):
        question = row['question']
        tool_list = convert_desc2format(row, self.method)
        tool_list_txt = '\n\n'.join(tool_list)
        task_spec = self.task_specification.format(tool_list=tool_list_txt)
        
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
            'question': 'Question: '+question+'\nAPI-Request:\n'
        }
        return self.template.render(data)


    def parse_answer(self, response, row):
        
        if '```' not in response:
            return {
                'correct': False,
                'function_correct': False,
                'argument_correct': False,
                'answer': row,
                'predict': None,
                'parsed_result': None,
                'parse_failed': 0,
                'response_non_json': 1
            }
        else:
            predict = response.replace('```json','```').split('```')[1]
        # print(predict)
        # print(payload['answer_num'])
        try:
            parsed_results = json.loads(predict)
        except json.JSONDecodeError:
            return {
                'correct': False,
                'function_correct': False,
                'argument_correct': False,
                'answer': row,
                'predict': None,
                'parsed_result': None,
                'parse_failed': 1,
                'response_non_json': 0
            }
        fn_correct = 0
        if 'api_name' in parsed_results:
            fn_correct = int(parsed_results['api_name'] == ground_truth['function_name'])
        params_name_correct = 0
        if 'params' in parsed_results:
            params = parsed_results['params']
            keys = set(params.keys())
            params_name_correct = int(keys == set(ground_truth['params'].keys()))
        return {
                'correct': params_name_correct & fn_correct,
                'function_correct': fn_correct,
                'argument_correct': params_name_correct,
                'answer': row,
                'predict': parsed_results,
                'parsed_result': parsed_results,
                'parse_failed': 0,
                'response_non_json': 0
            }


