import re
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList


# Define a stopping condition for generation
class SpecificStringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, input_len):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.input_len = input_len

    def __call__(self, input_ids, scores, **kwargs):
        current_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[self.input_len:]

        return any(stop_string in current_text for stop_string in self.stop_strings)


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = [
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None


def extract_ground_truth(text):
    return text.split('####')[-1].strip()




def eval_gsm8k_zero_shot(model, tokenizer, use_cot_prompt=True, use_majority_vote=False, n_votes=1, temp=0):
    print('\nLoading dataset...')
    dataset = load_dataset('gsm8k', "main", split='test')
    datasize = len(dataset)
    print('gsm8k test size:', datasize)

    # Define a stopping condition for generation
    generation_util = [
        "Q:",
        "</s>",
        "<|im_end|>"
    ]

    results = []
    for i in tqdm(range(datasize), desc='Evaluating'):
        example = dataset[i]
        if use_cot_prompt:
            input_text = "Q: {question}\nA: Let's think step by step.".format(question=example['question'])
        else:
            input_text = 'Q: ' + example['question'] + '\nA:'
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        ground_truth_answer = extract_ground_truth(example['answer'])

        # Define a stopping condition for generation
        stop_criteria = SpecificStringStoppingCriteria(tokenizer, generation_util, len(input_text))
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        model_answers = []
        if use_majority_vote:
            for _ in range(n_votes):
                with torch.no_grad():
                    outputs = model.generate(**inputs, temperature=temp, max_new_tokens=512, do_sample=True,
                                             pad_token_id=tokenizer.eos_token_id,
                                             stopping_criteria=stopping_criteria_list)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract the final answer from the model's output
                output_text = output_text.split("A:")[-1].strip()
                model_answer = extract_predicted_answer(output_text)
                model_answers.append({'text': output_text, 'numeric': model_answer})
        else:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,
                                         stopping_criteria=stopping_criteria_list)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split("A:")[-1].strip()
            model_answer = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': model_answer})

        numeric_answers = [ma['numeric'] for ma in model_answers]
        filtered_answers = [num for num in numeric_answers if num is not None]
        majority_answer = Counter(filtered_answers).most_common(1)[0][0] if filtered_answers else None

        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        results.append({
            'question': example['question'],
            'gold_answer_text': example['answer'],
            'model_answers_text': [ma['text'] for ma in model_answers],
            'extracted_model_answers': numeric_answers,
            'extracted_gold_answer': ground_truth_answer,
            'majority_answer': majority_answer,
            'correct': correct
        })

    cnt = 0
    for result in results:
        if result['correct']:
            cnt += 1
    total = len(results)
    print(f"gsm8k zero shot Accuracy: {cnt} / {total} = {cnt / total :.4f}")
