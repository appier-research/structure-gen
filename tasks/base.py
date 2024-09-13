import yaml
import json
import xml
from copy import copy
import xml.etree.ElementTree as ET
from jinja2 import Environment, FileSystemLoader
from .llm_parser import LLMParser

class BaseTextPrompter():
    def __init__(self, config_filename, num_shots=8) -> None:
        self.num_shots = num_shots
        file_loader = FileSystemLoader('./tasks/templates')
        env = Environment(loader=file_loader)
        self.template = env.get_template('base.txt')
        with open(config_filename, 'r') as file:
            self.config = yaml.safe_load(file)
        self.fewshots = self.config['fewshots']
        self.task_specification = self.config['task_specification']
        self.format_instruct = self.config['format_instruct']['text']

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                fewshot_text += "Question: {}\nAnswer: {}. The answer is {}.\n".format(
                    example['question'], example['response']['reason'], example['response']['answer']
                )

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question
        }

        return self.template.render(data)


class BaseJSONPrompter():

    def __init__(self, config_filename, num_shots=8) -> None:
        self.num_shots = num_shots
        file_loader = FileSystemLoader('./tasks/templates')
        env = Environment(loader=file_loader)
        self.template = env.get_template('base.txt')

        # read config filename
        # load format_instruct -> json
        # load fewshots
        # load task_specification
        with open(config_filename, 'r') as file:
            self.config = yaml.safe_load(file)
        self.fewshots = self.config['fewshots']
        self.task_specification = self.config['task_specification']
        self.format_instruct = self.config['format_instruct']['json']

        self.parser_prompt = None
        if 'parser_prompt' in self.config and 'json' in self.config['parser_prompt']:
            self.parser_prompt = self.config['parser_prompt']['json']
            self.parser = LLMParser(self.parser_prompt,
                                    method='claude',
                                    model_name='claude-3-haiku-20240307'
                                )

    def parse_answer(self, response, row):
        parse_failed = 0
        response_non_json = 0
        
        if self.parser_prompt is not None:
            original_response = response
            response = self.parser.parse(response)

        if '```' not in response and 'answer"' not in response:
            response_non_json += 1
            predict = '{"answer": null}'
        elif '```' in response:
            predict = response.replace('```json','```').split('```')[1]
        else:
            predict = response
        # print(payload['answer_num'])
        try:
            parsed_results = json.loads(predict)
        except json.JSONDecodeError as e:
            answer = None
            if '"answer": "' in predict:
                answer = predict.split('"answer": "')[-1].split('"')[0]
            elif '"final_answer": ' in predict:
                answer = predict.split('"final_answer": "')[-1].split('"')[0]
            if answer is None:
                parse_failed += 1
            parsed_results = {'answer': answer}
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'answer': None }
        if 'final_answer' in parsed_results:
            parsed_results['answer'] = parsed_results['final_answer']
        elif 'answer' not in parsed_results:
            parsed_results['answer'] = None
        # exact match with answer
        predict = parsed_results['answer']
        if isinstance(predict, str):
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}
        if self.parser_prompt is not None:
            additional_fields['original_response'] = original_response
            additional_fields['response'] = response

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
            'question': 'Question: '+question
        }

        return self.template.render(data)

def dictify(r,root=True):
    if root:
        return {r.tag : dictify(r, False)}
    d = copy(r.attrib)
    if r.text:
        d["_text"]=r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag]=[]
        d[x.tag].append(dictify(x,False))
    return d

class BaseXMLPrompter():

    def __init__(self, config_filename, num_shots=8) -> None:
        self.num_shots = num_shots
        file_loader = FileSystemLoader('./tasks/templates')
        env = Environment(loader=file_loader)
        self.template = env.get_template('base.txt')

        # read config filename
        # load format_instruct -> json
        # load fewshots
        # load task_specification
        with open(config_filename, 'r') as file:
            self.config = yaml.safe_load(file)
        self.fewshots = self.config['fewshots']
        self.task_specification = self.config['task_specification']
        self.format_instruct = self.config['format_instruct']['xml']
        self.parser_prompt = None
        if 'parser_prompt' in self.config and 'xml' in self.config['parser_prompt']:
            self.parser_prompt = self.config['parser_prompt']['xml']
            self.parser = LLMParser(self.parser_prompt,
                                    method='claude',
                                    model_name='claude-3-haiku-20240307'
                                )


    def parse_answer(self, response, row):
        if self.parser_prompt is not None:
            original_response = response
            response = self.parser.parse(response)

        parse_failed = 0
        response_non_xml = 0
        try:
            if '```' in response:
                predict = response.replace('```xml','```').split('```')[1].strip()
            elif '<root>' in response:
                predict = '<root>'+response.split('<root>', maxsplit=1)[-1]
            else:
                predict = response
        except IndexError:
            response_non_xml += 1
            predict = '<root><answer>None</answer></root>'
        parsed_results = {}
        try:
            parsed_results = ET.fromstring(predict)
        except xml.etree.ElementTree.ParseError as e:
            parsed_results = ET.fromstring("<root><answer>None</answer></root>")
            parse_failed += 1
        if len(parsed_results.findall('answer')) != 0:
            predict = parsed_results.findall('answer')[0].text
            if predict is not None:
                predict = predict.replace(',','').replace('$','').split(' ')[0]
        elif len(parsed_results.findall('final_answer')) != 0:
            predict = parsed_results.findall('final_answer')[0].text
            if predict is not None:
                predict = predict.replace(',','').replace('$','').split(' ')[0]
        else:
            predict = None
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}
        if self.parser_prompt is not None:
            additional_fields['original_response'] = original_response

        xml_dict = dictify(parsed_results)
        if len(parsed_results.findall('final_answer')) != 0:
            xml_dict['root']['answer'] = [{'_text': parsed_results.findall('final_answer')[0].text}]
        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': xml_dict,
            'parse_failed': parse_failed,
            'response_non_xml': response_non_xml,
            **additional_fields
        }

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                if 'reason' in example['response']:
                    fewshot_text += "Question: {}\nAnswer:\n```xml\n<root>\n  <nreasoning>{}</nreasoning>\n  <answer>{}</answer>\n</root>\n```\n".format(
                        example['question'], example['response']['reason'], example['response']['answer']
                    )
                else:
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


class BaseYAMLPrompter():

    def __init__(self, config_filename, num_shots=8) -> None:
        self.num_shots = num_shots
        file_loader = FileSystemLoader('./tasks/templates')
        env = Environment(loader=file_loader)
        self.template = env.get_template('base.txt')

        # read config filename
        # load format_instruct -> json
        # load fewshots
        # load task_specification
        with open(config_filename, 'r') as file:
            self.config = yaml.safe_load(file)
        self.fewshots = self.config['fewshots']
        self.task_specification = self.config['task_specification']
        self.format_instruct = self.config['format_instruct']['yaml']
        self.parser_prompt = None
        if 'parser_prompt' in self.config and 'yaml' in self.config['parser_prompt']:
            self.parser_prompt = self.config['parser_prompt']['yaml']
            self.parser = LLMParser(self.parser_prompt,
                                    method='claude',
                                    model_name='claude-3-haiku-20240307'
                                )

    def parse_answer(self, response, row):
        if self.parser_prompt is not None:
            original_response = response
            response = self.parser.parse(response)

        parse_failed = 0
        response_non_yml = 0

        try:
            if '```' in response:
                predict = response.replace('```yml','```').replace('```yaml', '```').split('```')[1]
            else:
                predict = response
        except IndexError as e:
            predict = 'reasoning: ""\nanswer: ""'
            response_non_yml += 1

        predict = predict.strip()

        if 'answer: ' not in response and 'final_answer: ' not in response:
            predict = 'reasoning: ""\nanswer: ""'
            response_non_yml += 1

        try:
            parsed_results = yaml.safe_load(predict)
        except (yaml.parser.ParserError, yaml.scanner.ScannerError):
            parsed_results = yaml.safe_load('reasoning: ""\nanswer: ""')
            parse_failed += 1

        if 'answer' in parsed_results:
            predict = str(parsed_results['answer'])
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        elif 'final_answer' in parsed_results:
            parsed_results['answer'] = parsed_results['final_answer']
            predict = str(parsed_results['final_answer'])
            predict = predict.replace(',','').replace('$','').split(' ')[0]
        else:
            predict = None
        predict = str(predict).lower()
        answer = row['answer'].split('####')[-1].strip()
        correct = predict == answer
        additional_fields = {}
        if self.parser_prompt is not None:
            additional_fields['original_response'] = original_response
            additional_fields['response'] = response

        return {
            'correct': correct,
            'answer': answer,
            'predict': predict,
            'parsed_result': parsed_results,
            'parse_failed': parse_failed,
            'response_non_yml': response_non_yml,
            **additional_fields
        }

    def prompt(self, row):
        question = row['question']
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            fewshot_text = 'Here are some examples:\n'
            for example in self.fewshots[:self.num_shots]:
                if 'reason' in example['response']:
                    fewshot_text += 'Question: {}\nAnswer:\n```yaml\nreasoning: {}\nanswer: "{}"\n```\n'.format(
                        example['question'], example['response']['reason'], example['response']['answer']
                    )
                else:
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