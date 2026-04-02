import random
import time

# random.seed(time.time())
# random.seed(993)
import datasets

import torch

from datasets import load_dataset


def get_arc_easy(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/ai2_arc", "ARC-Easy", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question"] + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_arc_challenge(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question"] + "choices:" + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_openbookqa(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/openbookqa", "main", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question_stem"] + "choices:" + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_winogrande(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/winogrande", "winogrande_xs", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "sentence: " + traindata[i]["sentence"] + "option1: " + traindata[i]["option1"] + "option2: "\
                   + traindata[i]["option2"] + "answer: " + traindata[i]["answer"]
            tokenized_sample = tokenizer(text, padding='max_length', max_length=seq_len, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_hellaswag(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "Rowan/hellaswag", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            ctx = traindata[i]["ctx_a"] + " " + traindata[i]["ctx_b"].capitalize()
            tokenized_sample = tokenizer(traindata[i]["activity_label"] + ": " + ctx, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_boolq(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "google/boolq", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = f"{traindata[i]['passage']}\nQuestion: {traindata[i]['question']}?"
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_piqa(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "ybisk/piqa", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]["goal"] + "choices:1." + traindata[i]["sol1"] + "2." + traindata[i]["sol2"]
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
        download_mode="reuse_cache_if_exists"
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train', download_mode="reuse_cache_if_exists"
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_ptb(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'ptb_text_only', 'penn_treebank', split='train', download_mode="reuse_cache_if_exists"
    )

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            if i not in history:
                history.append(i)
                break
        tokenized_samples.append(trainenc.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_wikitext2(tokenizer, n_samples, seq_len):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split='train',
                             download_mode="reuse_cache_if_exists")

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
            if i not in history:
                history.append(i)
                break
        tokenized_samples.append(trainenc.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_alpaca_cleaned(tokenizer, n_samples, seq_len):
    traindata = load_dataset("yahma/alpaca-cleaned", split='train', download_mode="reuse_cache_if_exists")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['instruction'] + traindata[i]['input'] + traindata[i]['output']
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_orca_dataset(tokenizer, n_samples, seq_len):
    seq_len = 1024
    """Helper to load and format Orca dataset."""
    dd = load_dataset("Open-Orca/SlimOrca", split="train").select(range(n_samples))
    processed = []

    for item in dd:
        idx = 0 if item['conversations'][0]["from"] == "human" else 1
        dialog = [
            {"role": "user", "content": item['conversations'][idx]['value']},
            {"role": "assistant", "content": item['conversations'][idx + 1]['value']},
        ]
        text = tokenizer.apply_chat_template(dialog, tokenize=False,
                                             chat_template=default_chat_template(tokenizer)['default'])
        inps = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
            truncation=True,
        ).input_ids
        processed.append(inps)

    return torch.cat(processed, dim=0)
    #     processed.append({"text": text})
    #
    # return datasets.Dataset.from_list(processed)


def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == "wikitext":
        return get_wikitext2(tokenizer, n_samples, seq_len)
    elif dataset == "ptb":
        return get_ptb(tokenizer, n_samples, seq_len)
    elif dataset == "orca":
        return get_orca_dataset(tokenizer, n_samples, seq_len)
    elif dataset == "mix":
        v = torch.tensor([1267., 10042., 3072., 1838., 500., 570., 299.])
        v = v / torch.sum(v)
        n_samples = (n_samples * v).type(torch.int32)
        print(n_samples)
        tokenized_samples = []

        tokenized_samples.append(get_winogrande(tokenizer, n_samples[0], seq_len))
        print("down")
        tokenized_samples.append(get_hellaswag(tokenizer, n_samples[1], seq_len))
        print("down")
        tokenized_samples.append(get_boolq(tokenizer, n_samples[2], seq_len))
        print("down")
        tokenized_samples.append(get_piqa(tokenizer, n_samples[3], seq_len))
        print("down")
        tokenized_samples.append(get_openbookqa(tokenizer, n_samples[4], seq_len))
        print("down")
        tokenized_samples.append(get_arc_easy(tokenizer, n_samples[5], seq_len))
        print("down")
        tokenized_samples.append(get_arc_challenge(tokenizer, n_samples[6], seq_len))
        print("down")
        return torch.cat(tokenized_samples, dim=0)
    elif "tune":
        return get_alpaca_cleaned(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError




PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "Cohere/Command-nightly": "https://huggingface.co/Cohere/Command-nightly/blob/main/tokenizer.json",
    },
}

# fmt: off
DEFAULT_SYSTEM_PROMPT = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."
DEFAULT_RAG_PREAMBLE = """## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""
# fmt: on

def default_chat_template(self):
    """
    Cohere Tokenizer uses <|START_OF_TURN_TOKEN|> and <|END_OF_TURN_TOKEN|> to indicate each turn in a chat.
    Additioanlly, to indicate the source of the message, <|USER_TOKEN|>, <|CHATBOT_TOKEN|> and <|SYSTEM_TOKEN|>
    for user, assitant and system messages respectively.

    The output should look something like:
    <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ preamble }}<|END_OF_TURN_TOKEN|><BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ How are you? }}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ I am doing well! }}<|END_OF_TURN_TOKEN|>

    Use add_generation_prompt to add a prompt for the model to generate a response:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    '<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'

    """
    default_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% elif USE_DEFAULT_PROMPT == true %}"
        "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% if system_message != false %}"  # Start with system message
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}"
        "{% endif %}"
        "{% for message in loop_messages %}"  # Loop over all non-system messages
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
        "{% endif %}"
    )

    flag = True
    if hasattr(self, "use_default_system_prompt"):
        flag = self.use_default_system_prompt
    default_template = default_template.replace(
        "USE_DEFAULT_PROMPT", "true" if flag else "false"
    )
    default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
    default_template = default_template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

    tool_use_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% endif %}"
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
        "{{ '# Safety Preamble' }}"
        "{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}"
        "{{ '\n\n# System Preamble' }}"
        "{{ '\n## Basic Rules' }}"
        "{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}"
        "{{ '\n\n# User Preamble' }}"
        "{{ '\n' + system_message }}"
        "{{'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n'}}"
        "{% for tool in tools %}"
        "{% if loop.index0 != 0 %}"
        "{{ '\n\n'}}"
        "{% endif %}"
        "{{'```python\ndef ' + tool.name + '('}}"
        "{% for param_name, param_fields in tool.parameter_definitions.items() %}"
        "{% if loop.index0 != 0 %}"
        "{{ ', '}}"
        "{% endif %}"
        "{{param_name}}: "
        "{% if not param_fields.required %}"
        "{{'Optional[' + param_fields.type + '] = None'}}"
        "{% else %}"
        "{{ param_fields.type }}"
        "{% endif %}"
        "{% endfor %}"
        '{{ \') -> List[Dict]:\n    """\'}}'
        "{{ tool.description }}"
        "{% if tool.parameter_definitions|length != 0 %}"
        "{{ '\n\n    Args:\n        '}}"
        "{% for param_name, param_fields in tool.parameter_definitions.items() %}"
        "{% if loop.index0 != 0 %}"
        "{{ '\n        ' }}"
        "{% endif %}"
        "{{ param_name + ' ('}}"
        "{% if not param_fields.required %}"
        "{{'Optional[' + param_fields.type + ']'}}"
        "{% else %}"
        "{{ param_fields.type }}"
        "{% endif %}"
        "{{ '): ' + param_fields.description }}"
        "{% endfor %}"
        "{% endif %}"
        '{{ \'\n    """\n    pass\n```\' }}'
        "{% endfor %}"
        "{{ '<|END_OF_TURN_TOKEN|>'}}"
        "{% for message in loop_messages %}"
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{{'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\'Action:\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        \"tool_name\": title of the tool in the specification,\n        \"parameters\": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>'}}"
        "{% if add_generation_prompt %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
        "{% endif %}"
    )
    default_tool_message = DEFAULT_RAG_PREAMBLE.replace("\n", "\\n").replace("'", "\\'")
    tool_use_template = tool_use_template.replace("DEFAULT_SYSTEM_MESSAGE", default_tool_message)

    rag_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% endif %}"
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
        "{{ '# Safety Preamble' }}"
        "{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}"
        "{{ '\n\n# System Preamble' }}"
        "{{ '\n## Basic Rules' }}"
        "{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}"
        "{{ '\n\n# User Preamble' }}"
        "{{ '\n' + system_message }}"
        "{{ '<|END_OF_TURN_TOKEN|>'}}"
        "{% for message in loop_messages %}"  # Loop over all non-system messages
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
        "{% endif %}"
        "{% endfor %}"
        "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}"
        "{{ '<results>' }}"
        "{% for document in documents %}"  # Loop over all non-system messages
        "{{ '\nDocument: ' }}"
        "{{ loop.index0 }}\n"
        "{% for key, value in document.items() %}"
        "{{ key }}: {{value}}\n"
        "{% endfor %}"
        "{% endfor %}"
        "{{ '</results>'}}"
        "{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
        "{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}"
        "{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}"
        "{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}"
        "{% if citation_mode=='accurate' %}"
        "{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}"
        "{% endif %}"
        "{{ 'Finally, Write \\'Grounded answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.' }}"
        "{{ '<|END_OF_TURN_TOKEN|>' }}"
        "{% if add_generation_prompt %}"
        "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
        "{% endif %}"
    )
    default_rag_message = DEFAULT_RAG_PREAMBLE.replace("\n", "\\n").replace("'", "\\'")
    rag_template = rag_template.replace("DEFAULT_SYSTEM_MESSAGE", default_rag_message)

    return {"default": default_template, "tool_use": tool_use_template, "rag": rag_template}
