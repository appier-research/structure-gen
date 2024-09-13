import json
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser

name_mapping = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
idx2name = [ 'O', 'PERSON', 'PERSON', 'ORGANIZATION', 'ORGANIZATION', 'LOCATION', 'LOCATION', 'MISC', 'MISC']

def calculate_f1_ner_conll2003(gold_entities, pred_entities):
    true_positives = len(set(gold_entities) & set(pred_entities))
    false_positives = len(pred_entities) - true_positives
    false_negatives = len(gold_entities) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1

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
                            "description": 'List of entity enum, each Entity should be a dictionary {"entity": entity, "word": word }'
                        }
                    },
                    "required": ["reason", "entities"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/conll23-t1-struct.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, parsed_results, row):
        parse_failed = 0
        response_non_json = 0
        if isinstance(parsed_results, str):
            try:
                parsed_results = json.loads(parsed_results)
            except json.decoder.JSONDecodeError:
                parse_failed += 1
                parsed_results = {'entities': [] }
        if not isinstance(parsed_results, dict):
            parse_failed += 1
            parsed_results = {'entities': [] }
        if 'entities' not in parsed_results:
            parsed_results['entities'] = []
        # exact match with entities
        predict = parsed_results['entities']
        pred_mapping = { row['entity']: row['word'] for row in predict }
        predict_ent = []
        for token in row['tokens']:
            if token in pred_mapping:
                predict_ent.append(mapping[token])
            else:
                predict_ent.append('O')
        
        answer = [ idx2name[answer_idx] for answer_idx in row['ner_tags']]
        
        correct = calculate_f1_ner_conll2003(answer, predict_ent)
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
        question = ' '.join(row['tokens'])
        if self.num_shots == 0:
            fewshot_text = ''
        else:
            # fewshot_text = 'Here are some examples:\n'
            # for example in self.fewshots[:self.num_shots]:
            #     fewshot_text += "Question: {}\nAnswer:\n```json\n{}\n```\n".format(
            #         example['question'], json.dumps(example['response'], indent=4)
            #     )
            fewshot_text = ''

        data = {
            'task_specification': self.task_specification,
            'fewshot_text': fewshot_text.strip(),
            'format_instruct': self.format_instruct,
            'question': 'Question: '+question,
            'tools': [self.schema]
        }

        return self.template.render(data), data

