import gc

import datasets
import torch
import torch.nn.functional as F
from colorama import Fore
from torch import nn
from torch.utils.data import Subset, Dataset
from tqdm import tqdm

from termcolor import colored

from torch.utils.data import DataLoader

from typing import Optional, Tuple


class ReplaceMePruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prune(self, args):
        model, tokenizer = self.model, self.tokenizer
        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = model.model.layers
        elif "opt" in args.base_model:
            layers = model.model.decoder.layers
        pruning_num = int(len(layers) * args.final_s)
        cosine_dist(model, tokenizer, args.dataset, batch_size=1, max_length=1024, dataset_size=args.num_examples,
                    num_layer=pruning_num, dataset_column="text", args=args)


@torch.no_grad()
def select(model, tokenizer, dataloader, pruning_num, max_length):
    device = model.device
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.model.decoder.layers

    score = torch.zeros((len(layers) - pruning_num + 1,), device=device)
    cnt = 128
    for batch in tqdm(
            dataloader, desc="Collecting hidden states"):
        batch_inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids
        cnt -= 1
        if cnt < 0: break
        for inps in batch_inputs:
            input_ids = inps.unsqueeze(0).to(device)
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            for i in range(len(score)):
                score[i] += (F.normalize(hidden_states[i][0], p=2, dim=1) *
                             F.normalize(hidden_states[i + pruning_num][0], p=2, dim=1)).sum(dim=1).mean(dim=0)

    for i in range(len(score)):
        score[i] = 1 - score[i] / len(dataloader)
    print(score)

    start_index = torch.argmin(score).item()
    pruning_layers = list(range(start_index, start_index + pruning_num))
    print(pruning_layers)

    model.config.use_cache = use_cache
    return start_index


def cosine_dist(
        model,
        tokenizer,
        dataset,
        dataset_column: str,
        batch_size: int,
        max_length: int,
        dataset_size: Optional[int] = None,
        dataset_subset: Optional[str] = "eval",
        diag: bool = False,
        loss: str = "cosine",
        thri: bool = False,
        two_vectors: bool = False,
        num_layer: int = 0,
        args=None
):
    hidden_size = model.config.hidden_size

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.named_parameters():
        param[1].requires_grad = False

    model.eval()
    dataloader = get_calib_dataloader(
        dataset,
        dataset_subset,
        dataset_column,
        dataset_size,
        batch_size,
        tokenizer
    )
    if len(args.remove_list) == 0:
        start_id = select(model, tokenizer, dataloader, num_layer, max_length)
    else:
        start_id = args.remove_list[0]
    torch.cuda.empty_cache()

    def save_mlp_activation(name):
        """Returns a hook function that saves the module output under the key 'name'."""

        def hook(module, input, output):
            # Detach to avoid keeping computation history
            mlp_activations[name] = output.detach()

        return hook

    opt_flag = False
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        opt_flag = True
        layers = model.model.decoder.layers

    hooks = []
    # for i, layer in enumerate(model.model.layers):
    #     hooks.append(layer.mlp.register_forward_hook(save_mlp_activation(f'layer_{i}_mlp')))
    bias = 0
    if not opt_flag:
        hooks.append(
            layers[start_id - 1].mlp.register_forward_hook(save_mlp_activation(f'layer_{start_id - 1}_mlp')))
        bias = torch.tensor([0])
        device = layers[start_id - 1].mlp.down_proj.weight.device
    else:
        hooks.append(
            layers[start_id - 1].fc2.register_forward_hook(save_mlp_activation(f'layer_{start_id - 1}_mlp')))
        bias = layers[start_id - 1].fc2.bias
        device = layers[start_id - 1].fc2.weight.device

    mlp_activations = {}
    a1 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )
    a2 = torch.empty(
        (dataset_size * max_length, model.config.hidden_size),
        dtype=torch.bfloat16,
        device='cpu'
    )

    cnt = 0
    for batch in tqdm(
            dataloader,
            desc=Fore.RED + "Gathering Activations" + Fore.RESET,
            dynamic_ncols=True,
            colour="red"
    ):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["output_hidden_states"] = True

        with torch.no_grad():
            outputs = model(**inputs)

        # hidden_states = outputs.hidden_states[1:]
        # hidden_states_mlp_list = [
        #     mlp_activations[f'layer_{i}_mlp'] for i in range(model.config.num_hidden_layers)
        # ]
        # hidden_states_mlp = hidden_states_mlp_list[start_id - 1]

        hidden_states = outputs.hidden_states
        hidden_states_mlp = mlp_activations[f'layer_{start_id - 1}_mlp']

        # Reshape activations
        # hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size).to(torch.float32)
        hidden_states_i = hidden_states[start_id]#.view(-1, hidden_size).to(torch.float32)
        hidden_states_n = hidden_states[start_id + num_layer]#.view(-1, hidden_size).to(torch.float32)

        a1_batch = hidden_states_mlp
        a2_batch = hidden_states_n.to(hidden_states_mlp.device) + hidden_states_mlp - hidden_states_i.to(hidden_states_mlp.device) - bias.to(hidden_states_mlp.device)
        if len(a1_batch.shape) == 3:
            new_size = a1_batch.shape[0] * a1_batch.shape[1]
        else:
            new_size = a1_batch.shape[0]
        a1[cnt:cnt + new_size] = a1_batch.view(-1, hidden_size).to(torch.float32)
        a2[cnt:cnt + new_size] = a2_batch.view(-1, hidden_size).to(torch.float32)

        cnt += new_size

        del hidden_states_mlp, hidden_states_i, hidden_states_n

    for hook in hooks:
        hook.remove()

    a1 = a1[:cnt]
    a2 = a2[:cnt]
    torch.cuda.empty_cache()

    transform = adam_method(a1, a2, a3=None, loss=loss, diag=diag, two_vectors=two_vectors,
                                thri=thri).to(device)

    torch.cuda.empty_cache()

    # Apply transformation
    if not opt_flag:
        layers[start_id - 1].mlp.down_proj.load_state_dict({
            "weight": (transform.T @ layers[start_id - 1].mlp.down_proj.weight.to(
                torch.float32)).to(torch.bfloat16)
        })
    else:
        layers[start_id - 1].fc2.load_state_dict({
            "weight": (transform.T @ layers[start_id - 1].fc2.weight.to(
                torch.float32)).to(torch.bfloat16), "bias": bias.to(torch.bfloat16)
        })

    del layers[start_id: start_id+num_layer]

    # Final cleanup
    del a1, a2

    gc.collect()
    torch.cuda.empty_cache()


def adam_method(
        a1: torch.Tensor,
        a2: torch.Tensor,
        a3: torch.Tensor = None,
        loss: str = "cosine",
        diag: bool = False,
        two_vectors: bool = False,
        thri: bool = False
) -> torch.Tensor:
    """Optimize transformation using Adam optimizer."""

    class ActivationDataset(Dataset):
        def __init__(self, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor):
            self.a1, self.a2, self.a3 = a1, a2, a3

        def __len__(self) -> int:
            return len(self.a1)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            attn = [-1]
            if self.a3 is not None:
                attn = self.a3[idx]
            return self.a1[idx], self.a2[idx], attn

    # Initialize model and optimizer
    if diag:
        transform = torch.ones(a1.shape[1], requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([transform], lr=1e-4)
    elif two_vectors:
        t1 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        t2 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([t1, t2], lr=1e-4)
    else:
        model = LowerTriangularLinear(a1.shape[1], a1.shape[1]).to("cuda") if thri \
            else nn.Linear(a1.shape[1], a1.shape[1], bias=False).to("cuda")
        if not thri:
            model.weight.data.copy_(torch.eye(a1.shape[1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Define loss functions
    def cosine_loss(XA: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    loss_fn = {
        "cosine": cosine_loss,
        "mse": nn.MSELoss(reduction='mean'),
        "elasticnet": lambda XA, Y: nn.MSELoss(reduction='mean')(XA, Y) + \
                                    0.09 * torch.norm(XA, p=1) + \
                                    0.045 * torch.norm(XA, p=2) ** 2
    }[loss]

    # Training loop
    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    with tqdm(range(10), desc="Optimizing Transformation") as pbar:
        for _ in pbar:
            for X, Y, Z in loader:
                optimizer.zero_grad()

                if diag:
                    XA = X.float().to("cuda") @ torch.diag(transform)
                elif two_vectors:
                    XA = X.float().to("cuda") @ (t1 @ t2.T)
                else:
                    XA = model(X.float().to("cuda"))
                if len(Z) != 1:
                    XA += Z.float().to("cuda")
                loss_val = loss_fn(XA, Y.float().to("cuda"))
                loss_val.backward()
                optimizer.step()

                pbar.set_postfix({f'{loss} Loss': colored(f'{loss_val.item():.4f}', 'green')})

    # Return appropriate transformation
    if diag:
        return torch.diag(transform).to(torch.float32)
    elif two_vectors:
        return (t1 @ t2.T).to(torch.float32)
    return model.triangular_weight.T.to(torch.float32) if thri else model.weight.T.to(torch.float32)


class LowerTriangularLinear(nn.Module):
    """Linear layer with lower triangular weight matrix."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        identity = torch.eye(min(input_size, output_size))
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.weight.data[:identity.size(0), :identity.size(1)] = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.tril(self.weight).t()


def get_calib_dataloader(
        dataset: str,
        dataset_subset: str,
        dataset_column: str,
        dataset_size: Optional[int],
        batch_size: int,
        tokenizer
) -> DataLoader:
    """Load and prepare calibration dataset."""
    dataset_handlers = {
        'HuggingFaceFW/fineweb': lambda: datasets.load_dataset(dataset, name='sample-10BT', split=dataset_subset),
        'allenai/c4': lambda: datasets.load_dataset(dataset, 'en', split=dataset_subset),
        'arcee-ai/sec-data-mini': lambda: datasets.load_dataset(dataset, split=dataset_subset),
        'wikitext': lambda: datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_subset),
        'Open-Orca/SlimOrca': lambda: _load_orca_dataset(dataset_size, tokenizer),
        # 'fineweb_and_orca': lambda: _load_mixed_dataset(dataset_size, dataset_subset, tokenizer)
    }

    if dataset not in dataset_handlers:
        raise ValueError(f"Dataset {dataset} not implemented")

    data = dataset_handlers[dataset]()
    if dataset_size:
        data = data.select(range(dataset_size))

    return DataLoader(data[dataset_column], batch_size=batch_size, shuffle=False, drop_last=True)


def _load_orca_dataset(size: int, tokenizer) -> datasets.Dataset:
    """Helper to load and format Orca dataset."""
    dd = datasets.load_dataset("Open-Orca/SlimOrca", split="train").select(range(size))
    processed = []

    for item in dd:
        idx = 0 if item['conversations'][0]["from"] == "human" else 1
        dialog = [
            {"role": "user", "content": item['conversations'][idx]['value']},
            {"role": "assistant", "content": item['conversations'][idx + 1]['value']},
        ]
        text = tokenizer.apply_chat_template(dialog, tokenize=False,
                                             chat_template=default_chat_template(tokenizer)['default'])
        processed.append({"text": text})

    return datasets.Dataset.from_list(processed)


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

