"""

Once we have some text results under logging directory we can study which LLM is most similar in answer parsing to gpt-4-turbo

"""
import os
import glob
import json
import random
import yaml
from tqdm import tqdm
import numpy as np
from itertools import combinations

def init_seed_data():
    for task in ['sports']:
        template_parser = list(glob.glob(f"tasks/templates/{task}-*.yaml"))[0]
        # open yaml
        print(template_parser)
        with open(template_parser, 'r') as f:
            template = yaml.safe_load(f)
        parser_prompt = template['parser_prompt']['text']

        rows = []
        for jsonl_filename in glob.glob(f"logging/{task}-*/text_*.jsonl"):
            with open(jsonl_filename, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    data['src'] = jsonl_filename
                    data['model_name'] = jsonl_filename.split('/')[-1].split('_')[-3]
                    data['parser_prompt'] = parser_prompt
                    rows.append(data)
        if len(rows) == 0:
            print('failed', task)
        else:
            random.shuffle(rows)
            samples = rows[:200]
            print(samples[0])
            with open(f'study_data/inputs_{task}_sample.jsonl', 'w') as fout:
                for row in samples:
                    fout.write(json.dumps(row)+'\n')

def prompt_others():
    from llms.gemini_vertex import Gemini
    from llms.oai_chat import OpenAIChat
    from llms.claude import ClaudeChat
    parser_llms = {
        'claude-3-haiku-20240307': ClaudeChat('claude-3-haiku-20240307'),
        'gpt-3.5-turbo': OpenAIChat('gpt-3.5-turbo-0125'),
        'gpt-4o-mini': OpenAIChat('gpt-4o-mini'),
        'gpt-4-turbo': OpenAIChat('gpt-4-turbo-2024-04-09'),
        'gemini-1.5-pro': Gemini('gemini-1.5-pro'),
        'gemini-1.5-flash': Gemini('gemini-1.5-flash'),
    }
    for model_name, llm in parser_llms.items():
        for study_text_parser_jsonl in glob.glob("study_data/inputs_*"):
            output_filename = study_text_parser_jsonl.replace('/inputs', '/outputs-'+model_name)
            if os.path.exists(output_filename):
                continue

            print(output_filename)
            with open(study_text_parser_jsonl, 'r') as f, \
                open(output_filename, 'w') as fout:
                for line in tqdm(f, total=200):
                    payload = json.loads(line)
                    response = payload['llm_info']['output']
                    parser_prompt = payload['parser_prompt']
                    res, res_info = llm(parser_prompt+'\n'+response+'\nAnswer:')
                    payload['new_model_parser_output'] = res
                    payload['new_model_parser_info'] = res_info
                    payload['new_model_name'] = model_name
                    fout.write(json.dumps(payload)+'\n')


def plot_results():
    from sklearn.metrics import cohen_kappa_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    prompt_others()
    models_label = {'llama-3-8b': []}
    for task in ['lastletter', 'gsm8k', 'task280', 'multifin', 'sports', 'ddxplus']:
        for idx, model_name in enumerate(glob.glob(f"study_data/outputs-*_{task}_sample.jsonl")):
            with open(model_name, 'r') as f:
                for line in f:
                    payload = json.loads(line)
                    if idx == 0:
                        models_label['llama-3-8b'].append(payload['predict'])
                    model_name = payload['new_model_name']
                    if model_name not in models_label:
                        models_label[model_name] = []
                    models_label[model_name].append(payload['new_model_parser_output'])
    
    # Calculate the cohen_kappa_score between all models
    model_names = list(models_label.keys())
    num_models = len(model_names)
    kappa_matrix = np.zeros((num_models, num_models))

    for (i, model1), (j, model2) in combinations(enumerate(model_names), 2):
        # print(len(models_label[model1]))
        print(len(models_label[model2]), model2)
        kappa = cohen_kappa_score(models_label[model1], models_label[model2])
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa  # The matrix is symmetric

    # Find the model with the highest average kappa score
    avg_kappa_scores = np.mean(kappa_matrix, axis=1)
    best_model_index = np.argmax(avg_kappa_scores)
    best_model = model_names[best_model_index]

    print("Cohen's Kappa Score Matrix:")
    print(kappa_matrix)
    print("\nAverage Kappa Scores:")
    for model, score in zip(model_names, avg_kappa_scores):
        print(f"{model}: {score:.4f}")
    print(f"\nModel with highest average agreement: {best_model}")
    
    # Plotting only the heatmap
    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(kappa_matrix, annot=True, cmap="YlGnBu", xticklabels=model_names, yticklabels=model_names)
    ax.set_xticklabels(model_names, rotation=20, ha='right')

    plt.tight_layout()

    # Save the plot
    plt.savefig('kappa_scores_heatmap.pdf', format="pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    init_seed_data()
    prompt_others()
    plot_results()
