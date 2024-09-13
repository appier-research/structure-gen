import os
import json
import glob
import yaml
from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from utils import get_llm, load_prompting_fn


task_specific_parser = {
    'gsm8k': "tasks/templates/gsm8k-t1-free.yaml",
    'lastletter': "tasks/templates/lastletter-t1-free.yaml",
}

def clean_prev_answer():
    llm = get_llm('claude-3-haiku-20240307', None)

    task = 'lastletter'
    for model in ['gpt', 'gemini', 'claude']:
        config_filename = task_specific_parser[task]
        with open(config_filename, 'r') as file:
            config = yaml.safe_load(file)

        for jsonl_filename in glob.glob(f"logging/{task}-t*-f*/*{model}*.jsonl"):
            method = jsonl_filename.split('/')[-1].split('_')[0]
            if '-free' in jsonl_filename:
                continue
            if method == 'text':
                continue

            parse_prompt = config['format_instruct'][method]
            if method == 'json':
                parser = BaseJSONPrompter(config_filename, 0)
            elif method == 'xml':
                parser = BaseXMLPrompter(config_filename, 0)
            elif method == 'yaml':
                parser = BaseYAMLPrompter(config_filename, 0)

            output_jsonl_filename = jsonl_filename.replace("logging/", "data/llm_clean/")
            dirs = '/'.join(output_jsonl_filename.split('/')[:3])
            os.makedirs(dirs, exist_ok=True)
            added = set()
            if os.path.exists(output_jsonl_filename):
                with open(output_jsonl_filename, 'r') as fcache:
                    for line in fcache:
                        added.add(json.loads(line)['id'])
            print(jsonl_filename)
            with open(jsonl_filename, 'r') as f:
                for idx, line in enumerate(tqdm(f)):
                    data = json.loads(line)
                    if idx in added:
                        continue
                    if idx > 150:
                        break
                    parse = data['parse_failed']
                    if 'response_non_'+method in data and data['response_non_'+method]:
                        parse = True
                    if parse:
                        response = data['llm_info']['output']
                        prompt = parse_prompt.strip()+'\n'+response
                        res, res_info = llm(prompt)
                        second_output = parser.parse_answer(res, data)
                        if second_output['predict'] is None or second_output['predict'] == 'None':
                            print(res)
                        data['stage2res'] = res_info
                        data['correct'] = second_output['correct']
                        data['predict'] = second_output['predict']
                        data['parsed_result'] = second_output['parsed_result']
                        data['parse_failed'] = second_output['parse_failed']
                        if 'response_non_'+method in second_output:
                            data['response_non_'+method] = second_output['response_non_'+method]
                        else:
                            data['response_non_'+method] = 0

                    data['id'] = idx
                    with open(output_jsonl_filename, 'a') as fout:
                        fout.write(json.dumps(data)+'\n')

if __name__ == "__main__":
    task_specific_parser = {
        'gsm8k': 'tasks/templates/gsm8k-t1-free.yaml',
        'lastletter': 'tasks/templates/lastletter-t1-free.yaml',
        'task280': 'tasks/templates/task280-t1-free1.yaml',
        'sports': 'tasks/templates/sports-t1-free.yaml',
        'multifin': 'tasks/templates/multifin-t3-free.yaml',
        'ddxplus': 'tasks/templates/ddxplus-t1-free.yaml',
        'shuffleobj': 'tasks/templates/shuffleobj-t1-free.yaml',
    }
    task = 'lastletter'
    for model in [ 'gpt-4o-mini-2024-07-18' ]:
        print(model)
        llm = get_llm(model, 'openai')
        config_filename = task_specific_parser[task]
        with open(config_filename, 'r') as file:
            config = yaml.safe_load(file)
        for jsonl_filename in glob.glob(f"logging/{task}-t*-f*/text*{model}*.jsonl"):
            print(jsonl_filename)
            if '-hybrid' in jsonl_filename:
                continue
            if '-free' in jsonl_filename:
                continue
            if 'shots_0' not in jsonl_filename:
                continue
            for method in ['json']:
                parse_prompt = config['format_instruct'][method]
                parser = load_prompting_fn(task, method)(num_shots=0, template_src=config_filename)
                parser.parser_prompt = None
                output_jsonl_filename = jsonl_filename.replace("logging/", "data/text-formating/")
                output_jsonl_filename = output_jsonl_filename.replace('text_', method+'_')
                dirs = '/'.join(output_jsonl_filename.split('/')[:3])
                os.makedirs(dirs, exist_ok=True)
                added = set()
                if os.path.exists(output_jsonl_filename):
                    with open(output_jsonl_filename, 'r') as fcache:
                        for line in fcache:
                            added.add(json.loads(line)['id'])
                accuracy = []
                text_acc = []
                with open(jsonl_filename, 'r') as f, tqdm() as pbar:
                    for idx, line in enumerate(f):
                        data = json.loads(line)
                        if idx in added:
                            continue
                        # assert len(data['llm_info']['logprobs'])
                        response = data['response']
                        # prompt = 'Parse out the final answer found in this following response in JSON, your answer should only contain JSON with { "answer": ... }\nResponse:\n'+'\n'+response
                        prompt = parse_prompt.strip() +'\n'+ response
                        res, res_info = llm(prompt)
                        second_output = parser.parse_answer(res, data)
                        if second_output['predict'] is None or second_output['predict'] == 'None':
                            print(res)
                        if task == 'shuffleobj':
                            if second_output['predict'] and isinstance(second_output['predict'], str) and second_output['predict'][0] in ['A', 'B', 'C', 'D', 'E', "F", "G"]:
                                second_output['predict'] = second_output['predict'][0]
                                second_output['correct'] = second_output['predict'] == data['answer']
                        data['src'] = jsonl_filename
                        data['stage2res'] = res_info
                        text_acc.append(data['correct'])
                        data['correct'] = second_output['correct']
                        data['predict'] = second_output['predict']
                        data['parsed_result'] = second_output['parsed_result']
                        data['parse_failed'] = second_output['parse_failed']
                        if 'response_non_'+method in second_output:
                            data['response_non_'+method] = second_output['response_non_'+method]
                        else:
                            data['response_non_'+method] = 0
                        accuracy.append(second_output['correct'])
                        pbar.set_description("[acc: {:.4f}, text_acc: {:.4f}]".format(np.mean(accuracy), np.mean(text_acc)))
                        pbar.update(1)
                        data['id'] = idx
                        with open(output_jsonl_filename, 'a') as fout:
                            fout.write(json.dumps(data)+'\n')
                        if idx > 2000:
                            break