import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_acc(file_match):
    accuracies = []
    errors = []
    for jsonl_filename in glob.glob(file_match):
        accuracy = []
        with open(jsonl_filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'The final answer' in data['parsed_result']:
                    data['predict'] = data['parsed_result'].split('The final answer is ')[-1].strip()
                    data['correct'] = data['predict'] == data['answer']
                if 'The answer is' in data['parsed_result']:
                    data['predict'] = data['parsed_result'].split('The answer is ')[-1].strip()
                    data['correct'] = data['predict'] == data['answer']

                if 'answer' in data and 'final_answer' in data and data['final_answer'] is None:
                    print(data['correct'])
                
                accuracy.append(data['correct'])
                if 'parse_failed' in data:
                    errors.append(data['parse_failed'])
                else:
                    errors.append(0)
                if len(accuracy) > 2000:
                    break
        if len(accuracy) <= 0.2 and  'text-format' in jsonl_filename:
            print(jsonl_filename)
        accuracies.append(np.mean(accuracy))
    return accuracies, errors


def check_text_format(response, row):
    if '**' in response:
        predict = response.split('**')[1]
        return predict.split('**')[0] == row['answer']
    return row['answer'] == response
def extract_task(dataset_task, shots, model_substr):
    methods = {}
    reasoning_lengths = {}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for jsonl_filename in glob.glob(f"logging/{dataset_task}-*t*/*.jsonl"):
        if shots not in jsonl_filename:
            continue
        if model_substr not in jsonl_filename:
            continue

        accuracy = []
        reasoning_length = []
        reasoning_mapping = {}
        method = jsonl_filename.split('/')[-1].split('_')[0]
        if 'hybrid' in jsonl_filename:
            method += '-hybrid'
        if 'free' in jsonl_filename:
            method += '-free'
            ingested_jsonl = jsonl_filename.replace('logging/', 'data/')
            if os.path.exists(ingested_jsonl):
                with open(ingested_jsonl, 'r') as f:
                    for idx, line in enumerate(f):
                        payload = json.loads(line)
                        reasoning_mapping[idx] = payload['reasoning'].replace('Output:','').strip()
        if 'structure' in jsonl_filename:
            method += '-structure'
        if method not in methods:
            methods[method] = []
            reasoning_lengths[method] = []
        dataset_with_prompt = jsonl_filename.split('/')[-2]
        task, form = dataset_with_prompt.split('-')[1:3]
        with open(jsonl_filename, 'r') as f:
            for line in f:
                reasoning = None
                data = json.loads(line)
                if 'The final answer' in data['parsed_result'] and isinstance(data['parsed_result'], str):
                    data['predict'] = data['parsed_result'].split('The final answer is ')[-1].strip()
                    data['correct'] = data['predict'] == data['answer']
                if 'The answer is' in data['parsed_result'] and isinstance(data['parsed_result'], str):
                    data['predict'] = data['parsed_result'].split('The answer is ')[-1].strip()
                    data['correct'] = data['predict'] == data['answer']

                if 'parse_failed' not in data:
                    accuracy.append(data['correct'])
                    continue
                if 'xml' in method:
                    if 'root' in data['parsed_result']:
                        # if '-free' in method:
                        #     print(data['parsed_result']['root']['reason'][0])
                        if 'reason' in data['parsed_result']['root'] and '_text' in data['parsed_result']['root']['reason'][0]:
                            reasoning = data['parsed_result']['root']['reason'][0]['_text']
                        if 'step_by_step_reasoning' in data['parsed_result']['root'] and '_text' in data['parsed_result']['root']['step_by_step_reasoning'][0]:
                            reasoning = data['parsed_result']['root']['step_by_step_reasoning'][0]['_text']
                    else:
                        reasoning = data['parsed_result']['reasoning']['_text']
                elif isinstance(data['parsed_result'], dict):
                    reasoning_key = [ key for key in data['parsed_result'].keys() if key != 'answer']
                    if len(reasoning_key) != 0:
                        reasoning_key = reasoning_key[0]
                        reasoning = data['parsed_result'][reasoning_key]
                        if isinstance(reasoning, list):
                            reasoning = ' '.join([ str(r) for r in reasoning])
                    else:
                        reasoning = ''
                if reasoning is not None and isinstance(reasoning, str):
                    reasoning_length.append(len(reasoning))
                if method == 'text':
                    correct = check_text_format(data['parsed_result'], data)
                else:
                    correct = data['correct']
                accuracy.append(correct)
        # print(jsonl_filename)
        methods[method].append(np.mean(accuracy))
        results[method][task][form].append({
                'accuracy': np.mean(accuracy)
            })
    # print(results)
    all_method_averages = {}
    for task in sorted(results['text'].keys()):
        row = [f"{task:^10}"]
        for method in ['text', 'json', 'yaml', 'xml', 'struct-structure']:
            accuracies = []
            for s in results[method][task].values():
                accuracies += [ k['accuracy'] for k in s]
            scores = np.mean(accuracies)*100
            if method not in all_method_averages:
                all_method_averages[method] = []
            print(results[method][task])
            all_method_averages[method] += accuracies
    # print(dataset_task)
    format_accuracies = {}
    for method, accuracies in methods.items():
        scores = np.mean(accuracies)*100
        format_accuracies[method] = accuracies
        std = np.std([ acc*100 for acc in accuracies])
        print(method, f"{scores:.2f} ({std:.1f})")
    return format_accuracies

tasks = [ 'lastletter', 'gsm8k', 'shuffleobj', 'ddxplus', 'sports', 'task280', 'multifin']
formats = ['text', 'struct-structure', 'json',  'json-free']
# use this for appendix
models = ['claude-3-haiku-20240307',  'llama-3-8b', 'gpt-3.5-turbo','gemini-1.5-flash', 'gemma2-9b-it', 'mistral-7b', 'gpt-4o-mini']

models = ['llama-3-8b', 'gpt-3.5-turbo','gemini-1.5-flash', 'gemma2-9b-it']

model2pretty = {
    'gemini-1.5-flash': 'Gemini 1.5 Flash',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
    'gpt-3.5-turbo': 'GPT 3.5 Turbo',
    'gemma2-9b-it': 'Gemma2 9B Instruct',
    'llama-3-8b': 'LLaMA 3 8B Instruct'
}
method2pretty = {
    'hard': 'JSON-mode',
    'soft': 'FRI (JSON)',
    '2-step': 'NL to Format',
    'text': 'NL'
}

models = ['claude-3-haiku-20240307',  'llama-3-8b', 'gpt-3.5-turbo','gemini-1.5-flash', 'gemma2-9b-it', 'mistral-7b', 'gpt-4o-mini']

task_results = {}
for task in tasks:
    task_results[task] = {}
    print(task)
    for model in models:
        print(model)
        text_filename = f'logging/{task}-t*-f*/text_{model}*shots_0.jsonl'
        text_acc, errors = extract_acc(text_filename)
        json_filename = f'logging/{task}-t*-f*/json_{model}*shots_0.jsonl'
        json_acc, errors = extract_acc(json_filename)
        mode_filename = f'logging/{task}-t*-structur*/struct_{model}*shots_0.jsonl'
        mode_acc, errors = extract_acc(mode_filename)
        two_stage_filename = f'data/text-formating/{task}-t*-f*/json_{model}*shots_0.jsonl'
        two_acc, errors = extract_acc(two_stage_filename)
        if len(mode_acc) == 0:
            mode_acc = [0]
        task_results[task][model] = {}
        task_results[task][model]['text'] = text_acc
        task_results[task][model]['hard'] = mode_acc
        task_results[task][model]['soft'] = json_acc
        task_results[task][model]['2-step'] = two_acc
        print(np.mean(text_acc), np.mean(mode_acc), np.mean(json_acc), np.mean(two_acc), len(two_acc), np.mean(errors))

models = ['llama-3-8b','gemma2-9b-it', 'gpt-3.5-turbo','gemini-1.5-flash']

fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
# fig.suptitle('Accuracy by Task and Format for Different Models', fontsize=16)
tasks = ['gsm8k', 'lastletter', 'shuffleobj']
# Set width of bars and positions
bar_width = 0.2
r = np.arange(len(models))

for idx, (ax, task) in enumerate(zip(axs, tasks)):
# # Set up the plot
# fig, ax = plt.subplots(figsize=(12, 6))
    # Set width of bars and positions of the bars on the x-axis
    bar_width = 0.2
    r = np.arange(len(models))
    
    # Plot bars for each format
    for i, fmt in enumerate(['hard', 'soft','2-step', 'text']):
        means = []
        stds = []
        for model in models:
            accuracies = task_results[task][model][fmt]
            means.append(np.mean(accuracies))
            # print(model, fmt, np.mean(accuracies))
            std_dev = np.std(accuracies, ddof=1)
            se = std_dev / np.sqrt(len(accuracies))
            stds.append(se)
        ax.bar(r + i*bar_width, means, bar_width, yerr=stds, capsize=5, 
               label=method2pretty[fmt], alpha=0.8)
    # Add labels and title
    if idx <= 1:
        ax.set_ylabel('EM (%)')
    else:
        ax.set_ylabel('Accuracy (%)')
    if task == 'lastletter':
        ax.set_title("Last Letter")
    if task == 'gsm8k':
        ax.set_title("GSM8K")
    if task == 'shuffleobj':
        ax.set_title("Shuffled Obj")
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_xticklabels([ model2pretty[m] for m in models ], rotation=20, ha='center')
    # Add legend
    if idx == 0:
        ax.legend()
    # Adjust layout and display the plot
plt.tight_layout()
plt.savefig("reasoning_comparison_restriction.pdf", format="pdf", bbox_inches="tight")



models = ['llama-3-8b','gemma2-9b-it', 'gpt-3.5-turbo','gemini-1.5-flash']
tasks = ['ddxplus', 'sports', 'task280', 'multifin']
# Set width of bars and positions
bar_width = 0.2
r = np.arange(len(models))
title_map = {
    'ddxplus': 'DDXPlus',
    'sports': "Sports",
    "task280": "NL Task 280",
    "multifin": 'MultiFin'
}
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True)

for idx, (ax, task) in enumerate(zip(axs, tasks)):
    bar_width = 0.2
    r = np.arange(len(models))
    
    # Plot bars for each format
    for i, fmt in enumerate(['hard', 'soft','2-step', 'text']):
        means = []
        stds = []
        for model in models:
            accuracies = task_results[task][model][fmt]
            means.append(np.mean(accuracies))
            std_dev = np.std(accuracies, ddof=1)
            se = std_dev / np.sqrt(len(accuracies))
            stds.append(se)
        ax.bar(r + i*bar_width, means, bar_width, yerr=stds, capsize=5, 
               label=method2pretty[fmt], alpha=0.8)
    # Add labels and title
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title_map[task])
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_xticklabels([ model2pretty[m] for m in models ], rotation=20, ha='right')
    
    # Add legend
    if idx == 3:
        ax.legend()
plt.tight_layout()
plt.savefig("classification_comparison_restriction.pdf", format="pdf", bbox_inches="tight")



models = ['llama-3-8b','gemma2-9b-it', 'gpt-3.5-turbo','gemini-1.5-flash']
tasks = ['ddxplus', 'sports', 'task280', 'multifin']
# Set width of bars and positions
bar_width = 0.2
r = np.arange(len(models))
title_map = {
    'ddxplus': 'DDXPlus',
    'sports': "Sports",
    "task280": "NL Task 280",
    "multifin": 'MultiFin'
}
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True)

for idx, (ax, task) in enumerate(zip(axs.flatten(), tasks)):
    bar_width = 0.2
    r = np.arange(len(models))
    
    # Plot bars for each format
    for i, fmt in enumerate(['hard', 'soft','2-step', 'text']):
        means = []
        stds = []
        for model in models:
            accuracies = task_results[task][model][fmt]
            means.append(np.mean(accuracies))
            std_dev = np.std(accuracies, ddof=1)
            se = std_dev / np.sqrt(len(accuracies))
            stds.append(se)
        ax.bar(r + i*bar_width, means, bar_width, yerr=stds, capsize=5, 
               label=method2pretty[fmt], alpha=0.8)
    # Add labels and title
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title_map[task])
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_xticklabels([ model2pretty[m] for m in models ], rotation=20, ha='right')
    
    # Add legend
    if idx == 3:
        ax.legend()
plt.tight_layout()
plt.savefig("classification_comparison_restriction_square.pdf", format="pdf", bbox_inches="tight")

tasks = ['lastletter', 'gsm8k', 'multifin', 'sports', 'task280', 'ddxplus', 'shuffleobj']
formats = ['text', 'json', 'yaml', 'xml']

# Plot bars for each format
all_data = {}
for model in ['gemini-1.5-flash', 'claude-3', 'gpt-3.5', 'gemma2-9b', 'llama-3', 'mistral-7b', 'gpt-4o-mini']:
    if model not in all_data:
        all_data[model] = {}
    for task in tasks:
        results = extract_task(task, 'shots_0', model)
        for i, fmt in enumerate(formats):
            accuracies = results[fmt]
            if task not in all_data[model]:
                all_data[model][task] = {}
            all_data[model][task][fmt] = accuracies

mapping = {'text': "Natural Language", 'json': "JSON", 'xml': "XML", 'yaml': "YAML"}

fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
# fig.suptitle('Accuracy by Task and Format for Different Models', fontsize=16)
tasks = ['gsm8k','lastletter', 'shuffleobj']
models = [
          ('Gemini 1.5 Flash', 'gemini-1.5-flash'),
          ('Claude 3.5 Haiku', 'claude-3'),
          ('GPT-3.5-Turbo', 'gpt-3.5'),
            ('LLaMA 3 8B', 'llama-3'),
          ('Gemma2 9B Instruct', 'gemma2-9b'),
         ]

# Set width of bars and positions
bar_width = 0.2
r = np.arange(len(models))
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
colors = ['#D5E5F2', '#C7CEEA', '#FFDAC1', '#E0C3FC']
colors = ['#2E86AB', '#F6F5AE', '#F5F749', '#F24236']
colors = ['#011f4b', '#03396c', '#005b96', '#6497b1']
colors = ['#1d3557', '#457b9d', '#a8dadc', '#00b4d8', '#0077be', '#48cae4']
for idx, (ax, task) in enumerate(zip(axs, tasks)):
    # Plot bars for each format
    for i, fmt in enumerate(formats):
        means = []
        stds = []
        for (model, model_task) in models:
            scores = all_data[model_task][task][fmt]
            scores = [src for src in scores if src > 0.05]
            mean_score = np.mean(scores)
            std_dev = np.std(scores, ddof=1)
            se = std_dev / np.sqrt(len(scores))
            means.append(mean_score)
            stds.append(se)
        if len(means):
            ax.bar(r + i*bar_width, means, bar_width, yerr=stds, capsize=5, 
                   label=mapping[fmt], alpha=0.8, color=colors[i])

    # Add labels
    if idx <= 1:
        ax.set_ylabel('EM (%)')
    else:
        ax.set_ylabel('Accuracy (%)')
    if task == 'lastletter':
        ax.set_title("Last Letter")
    if task == 'gsm8k':
        ax.set_title("GSM8K")
    if task == 'shuffleobj':
        ax.set_title("Shuffled Obj")

    # Set up the x-axis
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_xticklabels([m[0] for m in models], rotation=20, ha='center')
    
    # Add legend
    if idx == 0:
        ax.legend()

plt.savefig("reasoning_format_comparison_model.pdf", format="pdf", bbox_inches="tight")
plt.savefig("reasoning_format_comparison_model.jpg", format="jpeg", bbox_inches="tight")
plt.tight_layout()


models = [
          ('Gemini 1.5 Flash', 'gemini-1.5-flash'),
          ('Claude 3.5 Haiku', 'claude-3'),
          ('GPT-3.5-Turbo', 'gpt-3.5'),
            ('LLaMA 3 8B', 'llama-3'),
          ('Gemma2 9B Instruct', 'gemma2-9b'),
         ]

fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharex=True)
# fig.suptitle('Accuracy by Task and Format for Different Models', fontsize=16)
tasks = ['sports', 'task280', 'multifin', 'ddxplus']
tasks = ['ddxplus', 'sports', 'task280', 'multifin']
mapping = {'text': "NL", 'json': "JSON", 'xml': "XML", 'yaml': "YAML"}

# Set width of bars and positions
bar_width = 0.2
r = np.arange(len(models))
title_map = {
    'ddxplus': 'DDXPlus',
    'sports': "Sports",
    "task280": "NL Task 280",
    "multifin": 'MultiFin'
}
for idx, (ax, task) in enumerate(zip(axs, tasks)):
    # Plot bars for each format
    for i, fmt in enumerate(formats):
        means = []
        stds = []
        for (model, model_task) in models:
            scores = all_data[model_task][task][fmt]
            scores = [src for src in scores if src > 0.05]
            mean_score = np.mean(scores)
            std_dev = np.std(scores, ddof=1)
            se = std_dev / np.sqrt(len(scores))
            means.append(mean_score)
            stds.append(se)
        if len(means):
            ax.bar(r + i*bar_width, means, bar_width, yerr=stds, capsize=5, 
                   label=mapping[fmt], alpha=0.8, color=colors[i])
    # Add labels
    if idx == 0:
        ax.set_ylabel('Accuracy (%)')
    ax.set_title(title_map[task])
    ax.set_ylim(0, 0.9)

    # Set up the x-axis
    ax.set_xticks(r + bar_width * 1.5)
    ax.set_yticks(np.arange(0, 0.9, step=0.2))
    ax.set_xticklabels([m[0] for m in models], rotation=15, ha='right')
    # Add legend
    ax.legend()
plt.tight_layout()
plt.savefig("classification_format_comparison_model.pdf", format="pdf", bbox_inches="tight")
plt.savefig("classification_format_comparison_model.jpg", format="jpeg", bbox_inches="tight")