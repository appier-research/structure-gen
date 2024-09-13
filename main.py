import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_data_by_name, load_prompting_fn, get_llm

# Setting up argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Test LLMs on different datasets with various prompt styles.')
    parser.add_argument('--model', type=str, help='Model name, e.g., "gpt-3.5-turbo" or "gemini-1.0-pro"')
    parser.add_argument('--dataset', type=str, help='Dataset name, e.g., "gsm8k" or "ddxplus"')
    parser.add_argument('--series', type=str, default=None, help='Model API provider')
    parser.add_argument('--prompt_style', type=str, help='Prompt style, e.g., "yaml", "json", "xml"')
    parser.add_argument('--prompt_parser', type=None, help='Prompt function parser, if not provided use the same as prompt_style')
    parser.add_argument('--num_shots', type=int, default=8, help='number of few shot')
    parser.add_argument('--prefix', type=str, default='', help='additional prefix to add to your prompt style')
    parser.add_argument('--prompt_version', type=str, default=None, help='path to prompt yaml config, default is: tasks/template/<dataset>.yaml ')
    parser.add_argument('--batch', type=bool, default=False, help='Enable batch processing')
    parser.add_argument('--batch_id', type=str, default=None, help='Enable batch processing')
    return parser.parse_args()

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def process_dataset(model_name, dataset_name, prompt_style, prompt_version=None, model_series=None, num_shots=8, prefix=''):
    llm = get_llm(model_name=model_name, series=model_series)
    dataset = load_data_by_name(dataset_name)
    task_module = dataset_name
    if prompt_version is None:
        prompt_version  = 'tasks/templates/{}.yaml'.format(dataset_name)
        assert os.path.exists(prompt_version)
    else:
        basename = os.path.basename(prompt_version)
        task_module = os.path.splitext(basename)[0]
    # if 'structure' in task_module:
    #     task_module += str(1)
    results = []
    processed_ids = set()

    os.makedirs("logging", exist_ok=True)
    os.makedirs("logging/"+task_module, exist_ok=True)

    result_file = f"logging/{task_module}/{prefix}{prompt_style}_{model_name.split('/')[-1]}_shots_{num_shots}.jsonl"
    print(result_file)
    print(prompt_version)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])

    prompt_cls = load_prompting_fn(dataset_name, prompt_style)
    prompt_fn = prompt_cls(num_shots=num_shots, template_src=prompt_version)
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            if idx in processed_ids:
                pbar.update(1)
                continue

            prompt = prompt_fn.prompt(row)
            if isinstance(prompt, tuple):
                prompt, data = prompt
                tools = data['tools']
                response, res_info = llm(prompt, tools)
            else:
                # Default to plain text if no style specified
                response, res_info = llm(prompt)
            # i know its weird
            result = prompt_fn.parse_answer(response, row)
            results.append(result['correct'])

            with open(result_file, 'a') as fout:
                fout.write(json.dumps({
                    'question': row['question'],
                    'response': response,
                    'idx': idx,
                    'answer': row['answer'],
                    'llm_info': res_info,
                    **result
                }, default=set_default) + '\n')
            pbar.update(1)
            pbar.set_description("acc={:.4f}".format(np.mean(results)))

    accuracy = np.mean(results) if results else 0
    print(f"Accuracy for {model_name} on {dataset_name} using {prompt_style} prompts: {accuracy}")


def batch_inference(model_name, dataset_name, prompt_style, prompt_version=None, model_series=None, num_shots=8, prefix='', batch_id=False):
    assert 'gpt' in model_name
    from openai import OpenAI
    from time import sleep
    params = {'api_key': os.environ['OAI_KEY']}
    client = OpenAI(**params)
    dataset = load_data_by_name(dataset_name)
    task_module = dataset_name
    if prompt_version is None:
        prompt_version  = 'tasks/templates/{}.yaml'.format(dataset_name)
        assert os.path.exists(prompt_version)
    else:
        basename = os.path.basename(prompt_version)
        task_module = os.path.splitext(basename)[0]

    results = []
    processed_ids = set()
    os.makedirs("logging", exist_ok=True)
    os.makedirs("logging/"+task_module, exist_ok=True)
    os.makedirs("batch_cache", exist_ok=True)
    os.makedirs("batch_cache/"+task_module, exist_ok=True)

    result_file = f"logging/{task_module}/{prefix}{prompt_style}_{model_name.split('/')[-1]}_shots_{num_shots}.jsonl"
    batch_path = f"batch_cache/{task_module}/{prefix}{prompt_style}_{model_name.split('/')[-1]}_shots_{num_shots}-{task_module}.jsonl"
    print(result_file)
    print(prompt_version)

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['idx'])
                results.append(data['correct'])

    prompt_cls = load_prompting_fn(dataset_name, prompt_style)
    prompt_fn = prompt_cls(num_shots=num_shots, template_src=prompt_version)
    print('initialing batch file')
    customid2row = {}
    total_found = len(dataset)
    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar, open(batch_path, 'w') as fout:
        for idx, row in enumerate(dataset):
            if idx in processed_ids:
                continue

            prompt = prompt_fn.prompt(row)
            question_id = 'q{}-{}-{}-shots-{}'.format(idx, prefix, prompt_style, num_shots)
            customid2row[question_id] = row
            payload = {
                "custom_id": question_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model_name, "messages": [{"role": "user", "content": prompt }],"max_tokens": 1000}
            }
            if batch_id is None:
                fout.write(json.dumps(payload)+'\n')
            pbar.update(1)

    if len(processed_ids) == total_found:
        return None

    if batch_id is None:
        batch_input_file = client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        res = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{task_module}-{prefix}{prompt_style}_{model_name.split('/')[-1]}_shots_{num_shots}"
            }
        )
        batch_id = res.id
        prev_status = res.status
    else:
        prev_status = 'unknown'
    print("batch upload sucessfully, your job id is : {}".format(batch_id))
    with tqdm() as pbar:
        while True:
            sleep(5)
            res = client.batches.retrieve(batch_id)
            if res.status != prev_status:
                pbar.update(1)
                pbar.set_description('[{}->{}]'.format(prev_status, res.status))
            if res.status == 'completed':
                break
    output_batch_path = f"batch_cache/{task_module}/{prefix}{prompt_style}_{model_name.split('/')[-1]}_shots_{num_shots}_output.jsonl"
    output_file_id = res.output_file_id
    file_response = client.files.content(output_file_id)
    results = []
    print('writting to ', result_file)
    with tqdm(total=len(customid2row), dynamic_ncols=True) as pbar:
        for line in file_response.text.split('\n'):
            if len(line.strip()):
                payload = json.loads(line)
                custom_id = payload['custom_id']
                row = customid2row[custom_id]
                response = payload['response']['body']['choices'][0]['message']['content']
                res_info = payload['response']['body']['usage']
                result = prompt_fn.parse_answer(response, row)
                results.append(result['correct'])
                with open(result_file, 'a') as fout:
                    fout.write(json.dumps({
                        'question': row['question'],
                        'response': response,
                        'idx': idx,
                        'answer': row['answer'],
                        'llm_info': res_info,
                        **result
                    }, default=set_default) + '\n')
                pbar.update(1)
                pbar.set_description("acc={:.4f}".format(np.mean(results)))

# Main function to coordinate the processing
def main():
    args = parse_arguments()
    if args.prompt_parser:
        args.prompt_parser = args.prompt_style
    if args.batch:
        batch_inference(args.model, args.dataset, args.prompt_style,
                    model_series=args.series,
                    num_shots=args.num_shots,
                    prompt_version=args.prompt_version,
                    prefix=args.prefix,
                    batch_id=args.batch_id
                )
    else:
        process_dataset(args.model, args.dataset, args.prompt_style,
                        model_series=args.series,
                        num_shots=args.num_shots,
                        prompt_version=args.prompt_version,
                        prefix=args.prefix
                    )

if __name__ == "__main__":
    main()