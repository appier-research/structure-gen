import os
import glob
import json
import argparse
import numpy as np


def export_all_results():
    import pandas as pd
    results = []
    for dataset_name in ['ddxplus', 'lastletter', 'task280', 'gsm8k', 'multifin', 'sports']:
        for jsonl_filename in glob.glob(f"logging/{dataset_name}-*t*/*.jsonl"):
            model_name = jsonl_filename.split('_')[-3]
            accuracy = []
            format_error = []
            parsing_error = []
            reasoning_length = []
            print(jsonl_filename)
            method = jsonl_filename.split('/')[-1].split('_')[0]
            if 'hybrid' in jsonl_filename:
                method += '-hybrid'
            if 'structure' in jsonl_filename:
                method += '-struct'
            if 'free' in jsonl_filename:
                method += '-free'
            dataset_with_prompt = jsonl_filename.split('/')[-2]
            task, form = dataset_with_prompt.split('-')[1:3]
            num_shots = jsonl_filename.split('_')[-1].split('.')[0][-1]
            num_shots = int(num_shots)
            with open(jsonl_filename, 'r') as f:
                for line in f:
                    reasoning = None
                    data = json.loads(line)
                    if 'parse_failed' not in data:
                        accuracy.append(data['correct'])
                        continue
                    if method == 'xml':
                        if 'root' in data['parsed_result']:
                            if 'reason' in data['parsed_result']['root'] and '_text' in data['parsed_result']['root']['reason'][0]:
                                reasoning = data['parsed_result']['root']['reason'][0]['_text']
                            if 'step_by_step_reasoning' in data['parsed_result']['root'] and '_text' in data['parsed_result']['root']['step_by_step_reasoning'][0]:
                                reasoning = data['parsed_result']['root']['step_by_step_reasoning'][0]['_text']
                    elif isinstance(data['parsed_result'], dict):
                        reasoning_key = [ key for key in data['parsed_result'].keys() if key != 'answer']
                        if len(reasoning_key) != 0:
                            reasoning_key = reasoning_key[0]
                            reasoning = data['parsed_result'][reasoning_key]
                            if isinstance(reasoning, list):
                                reasoning = ' '.join([ str(r) for r in reasoning])
                            elif not isinstance(reasoning, str):
                                print(reasoning, 'failed')
                        else:
                            reasoning = ''
                    if reasoning is not None and isinstance(reasoning, str):
                        reasoning_length.append(len(reasoning))
                    accuracy.append(data['correct'])
                    res_key = None
                    for key in data.keys():
                        if 'response_non' in key:
                            res_key = key
                            break
                    format_error.append(data[res_key])
                    parsing_error.append(data['parse_failed'])
            
            reasoning_len = -1
            if len(reasoning_length) >= 100:
                reasoning_len = np.mean(reasoning_length)
            results.append({
                'dataset_name': dataset_name,
                'model_name': model_name,
                'method': method,
                'num_shots': num_shots,
                'accuracy': np.mean(accuracy),
                'total': len(accuracy),
                'parsing_error': np.mean(parsing_error),
                'format_error': np.mean(format_error),
                'reasoning_len': reasoning_len,
                'task': task,
                'format': form,
            })
    
    pd.DataFrame(results).to_csv("all_results.csv", index=False)

if __name__ == "__main__":
    # export_all_results()
    verbose = False
    parser = argparse.ArgumentParser(description='Test LLMs on different datasets with various prompt styles.')
    parser.add_argument('--src', type=str, help='Model name, e.g., "gpt-3.5-turbo" or "gemini-1.0-pro"')
    args = parser.parse_args()
    total_lines = -1
    for jsonl_filename in glob.glob(args.src+"/*.jsonl"):
        with open(jsonl_filename, 'r') as f:
            total_lines = max(len(f.readlines()), total_lines)


    aggregate = {}
    success_acc = {}
    parsing_error = {}
    format_error = {}
    for jsonl_filename in glob.glob(args.src+"/*.jsonl"):
        with open(jsonl_filename, 'r') as f:
            found_lines = len(f.readlines())
        # if found_lines != total_lines:
        #     continue

        tokens = jsonl_filename.split('/')[-1].replace('.jsonl','').split('_')
        model_name = tokens[1]
        num_shots = tokens[-1]
        if model_name not in aggregate:
            aggregate[model_name] = {}
            parsing_error[model_name] = {}
            format_error[model_name] = {}
            success_acc[model_name] = {}
        if num_shots not in aggregate[model_name]:
            aggregate[model_name][num_shots] = {}
            parsing_error[model_name][num_shots] = {}
            format_error[model_name][num_shots] = {}
            success_acc[model_name][num_shots] = {}
        method = '_'.join(tokens[:2])        
        accuracy = []
        success_accuracy = []
        parse_err = 0
        format_err = 0
        with open(jsonl_filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                accuracy.append(data['correct'])
                response_non_key = None
                for key in data.keys():
                    if 'response_non' in key:
                        response_non_key = key
                        break
                if response_non_key is not None:
                    format_err += data[response_non_key]
                    parse_err += data['parse_failed']
                    if (data[response_non_key]+data['parse_failed']) == 0:
                        success_accuracy.append(data['correct'])
        aggregate[model_name][num_shots][method] = sum(accuracy)/len(accuracy)
        if len(success_accuracy):
            success_acc[model_name][num_shots][method] = sum(success_accuracy)/len(success_accuracy)
        else:
            success_acc[model_name][num_shots][method] = 0
        parsing_error[model_name][num_shots][method] = parse_err/len(accuracy)
        format_error[model_name][num_shots][method] = format_err/len(accuracy)

    averaged_methods = {}
    for model_name, num_shots in aggregate.items():
        method_scores = {}
        for num_shot_, methods in num_shots.items():
            if verbose:
                print('-----------'*2)
                print(model_name, num_shot_)
            for key, accuracy in methods.items():
                token = key.split('/')[-1]
                token = token.split('_')[0]
                parse_err = parsing_error[model_name][num_shot_][key]
                format_err = format_error[model_name][num_shot_][key]
                success_ = success_acc[model_name][num_shot_][key]
                method = key.split('_')[0]
                if not verbose:
                    print(json.dumps({
                        'model_name': model_name,
                        'method': key.split('_')[0],
                        'num_shots': num_shot_,
                        'accuracy': accuracy,
                        'parse_err': parse_err,
                        'format_err': parse_err
                    }))
                if method not in averaged_methods:
                    averaged_methods[method] = {}
                if num_shot_ not in averaged_methods[method]:
                    averaged_methods[method][num_shot_] = []
                averaged_methods[method][num_shot_].append(accuracy)
            if verbose:
                print('-----------'*2)

    inverted = {}
    for method, shots_scores in averaged_methods.items():
        for num_shot, scores in shots_scores.items():
            if num_shot not in inverted:
                inverted[num_shot] = {}
            inverted[num_shot][method] = scores
    for num_shot, methods in inverted.items():
        print(num_shot)
        print('-------')
        for method, scores in methods.items():
            print(method, np.mean(scores), len(scores))
