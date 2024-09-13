import json
import re
import pandas as pd
from pydantic import BaseModel
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser
from .math_utils import ChatCompletionSampler, ANSWER_PATTERN, check_equality
from .normalizer import math_normalizer


class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/math-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.sampler = ChatCompletionSampler()

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
            predict = math_normalizer(predict)
        if predict is None:
            predict = parsed_results['answer']
        predict = str(predict)

        gold = math_normalizer(row['answer'])
        correct = float(check_equality(self.sampler, gold, predict))

        additional_fields = {}
        if self.parser_prompt is not None:
            additional_fields['original_response'] = original_response
            additional_fields['response'] = response

        return {
            'correct': correct,
            'answer': gold,
            'predict': predict,
            'parsed_result': parsed_results,
            'parse_failed': parse_failed,
            'response_non_json': response_non_json,
            **additional_fields
        }


class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/math-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/math-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)

class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=8, template_src='tasks/templates/math-t1-f1.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt, method='openai')
        self.sampler = ChatCompletionSampler()

    def parse_answer(self, response, row):
        """
        Extract the following response final answer, only number with no symbol no comma or full stop, only the numeric value. DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER NUMBER!
        Response: 
        <response>
        Answer:
        """
        if 'The answer is' in response:
            predict = math_normalizer(response.split("The answer is")[-1])
        elif 'Answer' in response:
            match = re.search(ANSWER_PATTERN, response)
            predict = match.group(1) if match else math_normalizer(response)
        else:
            predict = math_normalizer(self.parser.parse(response))

        gold = math_normalizer(row['answer'])
        correct = float(check_equality(self.sampler, gold, predict))
        predict = str(predict)

        return {
            'correct': correct,
            'answer': gold,
            'predict': predict,
            'parsed_result': predict
        }