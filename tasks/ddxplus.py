import json
from .base import BaseJSONPrompter, BaseXMLPrompter, BaseTextPrompter, BaseYAMLPrompter
from .llm_parser import LLMParser

LABEL2TEXT = {
    0: 'Acute COPD exacerbation / infection',
    1: 'Acute dystonic reactions',
    2: 'Acute laryngitis',
    3: 'Acute otitis media',
    4: 'Acute pulmonary edema',
    5: 'Acute rhinosinusitis',
    6: 'Allergic sinusitis',
    7: 'Anaphylaxis',
    8: 'Anemia',
    9: 'Atrial fibrillation',
    10: 'Boerhaave',
    11: 'Bronchiectasis',
    12: 'Bronchiolitis',
    13: 'Bronchitis',
    14: 'Bronchospasm / acute asthma exacerbation',
    15: 'Chagas',
    16: 'Chronic rhinosinusitis',
    17: 'Cluster headache',
    18: 'Croup',
    19: 'Ebola',
    20: 'Epiglottitis',
    21: 'GERD',
    22: 'Guillain-Barré syndrome',
    23: 'HIV (initial infection)',
    24: 'Influenza',
    25: 'Inguinal hernia',
    26: 'Larygospasm',
    27: 'Localized edema',
    28: 'Myasthenia gravis',
    29: 'Myocarditis',
    30: 'PSVT',
    31: 'Pancreatic neoplasm',
    32: 'Panic attack',
    33: 'Pericarditis',
    34: 'Pneumonia',
    35: 'Possible NSTEMI / STEMI',
    36: 'Pulmonary embolism',
    37: 'Pulmonary neoplasm',
    38: 'SLE',
    39: 'Sarcoidosis',
    40: 'Scombroid food poisoning',
    41: 'Spontaneous pneumothorax',
    42: 'Spontaneous rib fracture',
    43: 'Stable angina',
    44: 'Tuberculosis',
    45: 'URTI',
    46: 'Unstable angina',
    47: 'Viral pharyngitis',
    48: 'Whooping cough'
}
NOTINLABEL = len(LABEL2TEXT)
TEXT2LABEL = {v.lower(): k for k, v in LABEL2TEXT.items()}
LABEL_SET = {v.lower() for v in LABEL2TEXT.values()}

class JSONPrompter(BaseJSONPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/ddxplus.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        result['correct'] = result['answer'] == result['parsed_result']['answer']
        result['predict'] = result['parsed_result']['answer']
        return result

class XMLPrompter(BaseXMLPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/ddxplus.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        if 'root' in result['parsed_result'] and 'answer' in result['parsed_result']['root']:            
            result['correct'] = result['answer'] == result['parsed_result']['root']['answer'][0]['_text']
            result['predict'] = result['parsed_result']['root']['answer'][0]['_text']
        return result

class YAMLPrompter(BaseYAMLPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/ddxplus.yaml') -> None:
        super().__init__(template_src, num_shots)

    def parse_answer(self, response, row):
        result = super().parse_answer(response, row)
        # access the parsed_result here to calculate jaccard metrics
        if 'answer' in result['parsed_result']:
            result['correct'] = result['answer'] == result['parsed_result']['answer']
            result['predict'] = result['parsed_result']['answer']
        else:
            result['correct'] = False
            result['predict'] = None
        return result

class TextPrompter(BaseTextPrompter):
    def __init__(self, num_shots=2, template_src='tasks/templates/ddxplus.yaml') -> None:
        super().__init__(template_src, num_shots)
        self.parser_prompt = self.config['parser_prompt']['text']
        self.parser = LLMParser(self.parser_prompt,
                                method='openai',
                                model_name='claude-3-haiku-20240307'
                            )

    def parse_answer(self, response, row):
        parser_rule = False
        if 'The answer is' in response:
            predict = response.split("The answer is")[-1]
            predict = predict.strip()
            parser_rule = True
        else:
            predict = self.parser.parse(response)

        predict = predict.replace('*','').replace('#','').strip().replace('.','')
        if 'The final answer is:' in predict:
            predict = predict.split("The final answer is:")[-1]
            predict = predict.strip()
            parser_rule = True
        elif 'The final answer' in predict or 'answer' in predict.lower():
            predict = self.parser.parse(response)

        # access the parsed_result here to calculate jaccard metrics
        correct = row['answer'] == predict
        return {
            'correct': correct,
            'answer': row['answer'],
            'predict': predict,
            'parser_rule': parser_rule,
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
    def __init__(self, num_shots=8, template_src='tasks/templates/ddxplus.yaml') -> None:
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
                    "answer": {
                        "type": "string",
                        "enum": ["Possible NSTEMI / STEMI", "Spontaneous rib fracture","Pulmonary embolism","Pulmonary neoplasm","URTI","Croup","Sarcoidosis","HIV (initial infection)","Inguinal hernia","Spontaneous pneumothorax","Bronchospasm / acute asthma exacerbation","Viral pharyngitis","Bronchiolitis","Pancreatic neoplasm","Guillain-Barré syndrome","Chagas","Allergic sinusitis","Acute rhinosinusitis","PSVT","Panic attack","Epiglottitis","Bronchiectasis","Bronchitis","Pericarditis","Acute COPD exacerbation / infection","Ebola","Chronic rhinosinusitis","Acute otitis media","Larygospasm","Influenza","Stable angina","Myasthenia gravis","Myocarditis","SLE","GERD","Anemia","Cluster headache","Localized edema","Anaphylaxis","Atrial fibrillation","Acute pulmonary edema","Acute laryngitis","Acute dystonic reactions","Boerhaave","Pneumonia","Tuberculosis","Whooping cough","Unstable angina","Scombroid food poisonin"],
                        "description": "patient most diagnosis"
                    }
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
                "name": "get_diagnosis",
                "description": "Answer to the last question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step_by_step": {
                            "type": "string",
                            "description": "think step by step"
                        },
                        "final_answer": {
                            "type": "string",
                            "enum": ["Possible NSTEMI / STEMI", "Spontaneous rib fracture","Pulmonary embolism","Pulmonary neoplasm","URTI","Croup","Sarcoidosis","HIV (initial infection)","Inguinal hernia","Spontaneous pneumothorax","Bronchospasm / acute asthma exacerbation","Viral pharyngitis","Bronchiolitis","Pancreatic neoplasm","Guillain-Barré syndrome","Chagas","Allergic sinusitis","Acute rhinosinusitis","PSVT","Panic attack","Epiglottitis","Bronchiectasis","Bronchitis","Pericarditis","Acute COPD exacerbation / infection","Ebola","Chronic rhinosinusitis","Acute otitis media","Larygospasm","Influenza","Stable angina","Myasthenia gravis","Myocarditis","SLE","GERD","Anemia","Cluster headache","Localized edema","Anaphylaxis","Atrial fibrillation","Acute pulmonary edema","Acute laryngitis","Acute dystonic reactions","Boerhaave","Pneumonia","Tuberculosis","Whooping cough","Unstable angina","Scombroid food poisonin"],
                            "description": "patient most diagnosis"
                        },
                    },
                    "required": ["step_by_step", "final_answer"]
                }
            }
        }

    def __init__(self, num_shots=8, template_src='tasks/templates/ddxplus.yaml') -> None:
        super().__init__(template_src, num_shots)

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
        # exact match with answer
        predict = parsed_results['final_answer']
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