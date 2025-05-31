import os
import json
import numpy as np

def get_llm(model_name: str, series: str = None):
    if series is None:
        if model_name[:3] in ('gpt', 'o1-'):
            series = 'openai'
        elif model_name[:6] == 'claude':
            series = 'anthropic'
        elif model_name[:6] == 'gemini':
            series = 'gemini'
        if series is None:
            raise ValueError('unable to found matching series for current model, please provide the API provider in --series value')
    if series == 'gemini':
        from llms.gemini_vertex import Gemini
        return Gemini(model_name)
    elif series == 'structure':
        if 'gemini' in model_name:
            from llms.gemini_vertex_structure import GeminiStructure
            return GeminiStructure(model_name)
        from llms.oai_structure import OpenAIJSON, OpenAIChat
        if 'gpt' in model_name:
            return OpenAIChat(model_name)
        # TGI
        return OpenAIJSON(model_name)
    elif series == 'struct-v2':
        from llms.oai_structurev2 import OpenAIStructureV2
        if 'gpt' in model_name:
            return OpenAIStructureV2(model_name)
        raise ValueError("Deterministic JSON mode doesn't support this model: ", model_name)
    elif series == 'gemini_dev':
        from llms.gemini_dev import GeminiDev
        return GeminiDev(model_name)
    elif series == 'openai':
        from llms.oai_chat import OpenAIChat
        return OpenAIChat(model_name)
    elif series == 'anthropic':
        from llms.claude import ClaudeChat
        return ClaudeChat(model_name)
    elif series == 'anthropic_vertex':
        from llms.vertex_claude import ClaudeChat
        return ClaudeChat(model_name)
    elif series == 'hf_model':
        from llms.hf_model import HFModel
        return HFModel(model_name)
    elif series == 'outlines':
        from llms.outlines_model import OutlinesStructure
        return OutlinesStructure(model_name)
    elif series == 'xgrammar':
        from llms.xgrammar_model import XGrammar
        return XGrammar(model_name)
    elif series == 'tgi':
        from llms.tgi_grammar_model import TGI
        return TGI(model_name)
    elif series == 'sglang':
        from llms.sglang_model import SGLang
        return SGLang(model_name)
    elif series == "groq":
        from llms.groq_model import GroqModel
        return GroqModel(model_name)
    elif series == "together":
        from llms.together_model import TogetherModel
        return TogetherModel(model_name)
    raise ValueError('series : {} for {} is not yet supported'.format(series, model_name))



def load_data_by_name(task):
    from datasets import load_dataset, Dataset

    if task == 'gsm8k':
        return load_dataset('gsm8k', 'main', split='test')
    elif task == 'math':
        data = []
        for row in load_dataset("appier-ai-research/robust-finetuning", "math")['test']:
            row = dict(row)
            row['question'] = row['problem']
            row['answer'] = row['solution']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'ddxplus':
        data = []
        for row in load_dataset('appier-ai-research/StreamBench',
                            "ddxplus",
                            split='test'
                        ):
            row = dict(row)
            row['question'] = row['PATIENT_PROFILE']
            row['answer'] = row['PATHOLOGY']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'lastletter':
        return load_dataset('ChilleD/LastLetterConcat', split='test')
    elif task == 'multifin':
        data = []
        for row in load_dataset('ChanceFocus/flare-multifin-en',
                            split='test'
                        ):
            row = dict(row)
            row['question'] = row['text']
            row['answer'] = row['answer'].replace('&', 'and')
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'multiarith':
        data = []
        for row in load_dataset('ChilleD/MultiArith', split='test'):
            row = dict(row)
            row['question'] = row['question']
            row['answer'] = row['final_ans']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'shuffleobj':
        data = []
        choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for row in load_dataset('tasksource/bigbench', 'tracking_shuffled_objects', split='validation', trust_remote_code=True):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(choices, row['multiple_choice_targets'])])
            row['question'] = row['inputs']+'\n'+answer_choice
            row['answer'] = choices[np.argmax(row['multiple_choice_scores'])]
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'dateunder':
        data = []
        choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        for row in load_dataset('tasksource/bigbench', 'date_understanding', split='validation', trust_remote_code=True):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(choices, row['multiple_choice_targets'])])
            row['question'] = row['inputs']+'\n'+answer_choice
            row['answer'] = choices[np.argmax(row['multiple_choice_scores'])]
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'csqa':
        data = []
        for row in load_dataset('tau/commonsense_qa', split='validation'):
            row = dict(row)
            answer_choice = '\n'.join([ '{}) {}'.format(l, t) for l, t in zip(row['choices']['label'], row['choices']['text'])])
            row['question'] = row['question']+'\n'+answer_choice
            row['answer'] = row['answerKey']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'sports':
        data = []
        for row in load_dataset('tasksource/bigbench',
                                'sports_understanding',
                            split='validation',
                            trust_remote_code=True
                        ):
            row = dict(row)
            row['question'] = row['inputs'].replace('Determine whether the following statement or statements are plausible or implausible:','').replace('Statement: ','').replace('Plausible/implausible?','').strip()
            row['answer'] = 'yes' if row['targets'][0] == 'plausible' else 'no'
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'task280':
        with open('data/task280_stereoset_classification_stereotype_type.json', 'r') as f:
            raw_data = json.load(f)
        data = []
        for row in raw_data['Instances'][:1000]:
            data.append({
                'question': row['input'],
                'answer': row['output'][0].lower()
            })
        return Dataset.from_list(data)
    elif task == 'conll2003':
        data = []
        for row in load_dataset("eriktks/conll2003", split="test"):
            row = dict(row)
            question = ' '.join(row['tokens'])
            row['question'] = question
            row['answer'] = row['ner_tags']
            data.append(row)
        return Dataset.from_list(data)
    elif task == 'api-bank':
        data = []
        with open('API-Bank/test.jsonl', 'r') as f:
            for line in f:
                payload = json.loads(line)
                data.append(payload)
        return data

    raise ValueError("%s is not in supported list" % task)

def load_prompting_fn(task, prompt_style):
    if task == 'gsm8k':
        from tasks.gsm8k import (
            JSONPrompter,
            XMLPrompter,
            TextPrompter,
            YAMLPrompter,
            StructJSONPrompter,
            OAIStructPrompter,
            TwoStageLLM
        )
        formatter = {
            'json': JSONPrompter,
            'yaml': YAMLPrompter,
            'xml': XMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter,
            'struct-v2': OAIStructPrompter,
            '2stage': TwoStageLLM,
        }
    if task == 'math':
        from tasks.math import TextPrompter, YAMLPrompter, XMLPrompter, JSONPrompter
        formatter = {
            'json': JSONPrompter,
            'yaml': YAMLPrompter,
            'xml': XMLPrompter,
            'text': TextPrompter
        }
    elif task == 'multiarith':
        from tasks.gsm8k import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'yaml': YAMLPrompter,
            'xml': XMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'ddxplus':
        from tasks.ddxplus import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter, TwoStageLLM
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter,
            '2stage': TwoStageLLM
        }
    elif task == 'multifin':
        from tasks.multifin import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'csqa':
        from tasks.multifin import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter
        from tasks.csqa import StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter 
        }
    elif task == 'shuffleobj':
        from tasks.multifin import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter
        from tasks.shuffleobj import StructJSONPrompter, OAIStructPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter,
            'struct-v2': OAIStructPrompter
        }
    elif task == 'dateunder':
        from tasks.multifin import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter
        from tasks.dateunder import StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'lastletter':
        from tasks.lastletter import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter, OAIStructPrompter, TwoStageLLM
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter,
            'struct-v2': OAIStructPrompter,
            '2stage': TwoStageLLM,
        }
    elif task == 'sports':
        from tasks.sports import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'task280':
        from tasks.task280 import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'conll2003':
        from tasks.conll import JSONPrompter, XMLPrompter, TextPrompter, YAMLPrompter, StructJSONPrompter
        formatter = {
            'json': JSONPrompter,
            'xml': XMLPrompter,
            'yaml': YAMLPrompter,
            'text': TextPrompter,
            'struct': StructJSONPrompter
        }
    elif task == 'api-bank':
        from tasks.lastletter import JSONPrompter
        # from
        formatter = {
            'json': JSONPrompter
        }
    return formatter[prompt_style]

