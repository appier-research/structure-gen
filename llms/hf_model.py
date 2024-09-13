import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFModel():

    def __init__(self, model_name) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map='cuda' if torch.cuda.is_available() else 'cpu',
                                                            torch_dtype=torch.float16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, prompt, max_tokens=512, temperature=0.0, **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            outputs = self.model.generate(input_tensor.to(self.model.device),
                                          max_new_tokens=int(max_tokens),
                                          temperature=float(temperature)
                                        )

            result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            logging.error('anthropic:'+str(e))
            result = 'error:{}'.format(e)
        return result

    def _get_hint_distribution(self, prompt):
        yes_answer = prompt+' Answer YES'
        no_answer = prompt+' Answer NO'
        prefix_length = len(self.tokenizer(prompt).input_ids)
        yes_inputs = self.tokenizer(yes_answer, return_tensors='pt').to(self.model.device)
        postfix_idx = yes_inputs.input_ids[0, prefix_length:]
        with torch.no_grad():
            outputs = self.model(yes_inputs.input_ids, labels=yes_inputs.input_ids.clone())
        yes_values = []
        for idx, logits in zip(postfix_idx, outputs.logits[0, prefix_length:]):
            yes_values.append(logits[idx])
        yes_values = torch.stack(yes_values, dim=0)
        no_inputs = self.tokenizer(no_answer, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model(no_inputs.input_ids, labels=no_inputs.input_ids.clone())
        no_logits = outputs.logits[0, prefix_length:]
        no_values = []
        for idx, logits in zip(postfix_idx, no_logits):
            no_values.append(logits[idx])
        no_values = torch.stack(no_values, dim=0)
        probs = torch.softmax(torch.stack([yes_values, no_values], dim=0).mean(-1).unsqueeze(0), dim=-1)
        return {
            'yes': float(probs[0][0]),
            'no': float(probs[0][1])
        }

    def get_hint_distribution(self, prompt):
        if len(self.tokenizer('YES').input_ids) == 2 and len(self.tokenizer('NO').input_ids) == 2:
            logits_map = { 'yes': self.tokenizer('YES').input_ids[-1], 'no': self.tokenizer('NO').input_ids[-1] }
            prefix_length = len(self.tokenizer(prompt+' Answer').input_ids)
            inputs = self.tokenizer(prompt+' Answer', return_tensors='pt').to(self.model.device)
            results = {}
            with torch.no_grad():
                outputs = self.model(inputs.input_ids).logits[:, -1, :]
                probs = torch.softmax(outputs, dim=-1)
            for answer, logit in logits_map.items():
                results[answer] = float(probs[0][logit])

            return results

        yes_answer = prompt+' Answer YES'
        no_answer = prompt+' Answer NO'
        prefix_length = len(self.tokenizer(prompt+' Answer').input_ids)
        yes_inputs = self.tokenizer(yes_answer, return_tensors='pt').to(self.model.device)
        postfix_idx = yes_inputs.input_ids[0, prefix_length:]
        with torch.no_grad():
            outputs = self.model(yes_inputs.input_ids, labels=yes_inputs.input_ids.clone())
        yes_values = []
        for idx, logits in zip(postfix_idx, outputs.logits[0, prefix_length:]):
            yes_values.append(logits[idx])
        yes_values = torch.stack(yes_values, dim=0)
        no_inputs = self.tokenizer(no_answer, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model(no_inputs.input_ids, labels=no_inputs.input_ids.clone())
        no_logits = outputs.logits[0, prefix_length:]
        no_values = []
        for idx, logits in zip(postfix_idx, no_logits):
            no_values.append(logits[idx])
        no_values = torch.stack(no_values, dim=0)
        probs = torch.softmax(torch.stack([yes_values, no_values], dim=0).mean(-1).unsqueeze(0), dim=-1)

        return {
            'yes': float(probs[0][0]),
            'no': float(probs[0][1])
        }

if __name__ == "__main__":
    text = "You are doing text-to-SQL task.\nBased on the below information, you are tasked to reason whether you can detrmine if the answered SQL needed more hint to answer more correctly.\nNote only asked for hint if its actually incorrect, cause you will be graded based on the numbers of times you asked for hint but the answer is correct.Background:\nCREATE TABLE players(\n    \"player_id\" INT PRIMARY KEY,\n    \"first_name\" TEXT,\n    \"last_name\" TEXT,\n    \"hand\" TEXT,\n    \"birth_date\" DATE,\n    \"country_code\" TEXT\n)\n\nCREATE TABLE matches(\n  \"best_of\" INT,\n  \"draw_size\" INT,\n  \"loser_age\" FLOAT,\n  \"loser_entry\" TEXT,\n  \"loser_hand\" TEXT,\n  \"loser_ht\" INT,\n  \"loser_id\" INT,\n  \"loser_ioc\" TEXT,\n  \"loser_name\" TEXT,\n  \"loser_rank\" INT,\n  \"loser_rank_points\" INT,\n  \"loser_seed\" INT,\n  \"match_num\" INT,\n  \"minutes\" INT,\n  \"round\" TEXT,\n  \"score\" TEXT,\n  \"surface\" TEXT,\n  \"tourney_date\" DATE,\n  \"tourney_id\" TEXT,\n  \"tourney_level\" TEXT,\n  \"tourney_name\" TEXT,\n  \"winner_age\" FLOAT,\n  \"winner_entry\" TEXT,\n  \"winner_hand\" TEXT,\n  \"winner_ht\" INT,\n  \"winner_id\" INT,\n  \"winner_ioc\" TEXT,\n  \"winner_name\" TEXT,\n  \"winner_rank\" INT,\n  \"winner_rank_points\" INT,\n  \"winner_seed\" INT,\n  \"year\" INT,\n  FOREIGN KEY(loser_id) REFERENCES players(player_id),\n  FOREIGN KEY(winner_id) REFERENCES players(player_id)\n)\n\nCREATE TABLE rankings(\"ranking_date\" DATE,\"ranking\" INT,\"player_id\" INT,\"ranking_points\" INT,\"tours\" INT,FOREIGN KEY(player_id) REFERENCES players(player_id))\n\nQuesion : what are the names of the tourneys?\nAnswer :\n```sql\nSELECT tourney_name FROM matches;\n```\n\nThe executed result:\n```\nargument 1 must be str, not None\n```\nAnswer YES if you needed hint as the current information is not enough to answer this correctly\nAnswer NO if hint is not required as you already have the information.\nIf answered YES a hint will be given by human to help you better answer the question.\nThink step by step first"
    agent = HFModel('openbmb/MiniCPM-2B-sft-bf16')
    res = agent.get_hint_distribution(text)
    print(res)