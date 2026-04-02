import re

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers import (
    LlamaForCausalLM
)

from transformers.modeling_outputs import CausalLMOutputWithPast


class OnOff_LlamaDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.hidden_size = original_decoder_layer.hidden_size

        self.self_attn = original_decoder_layer.self_attn
        self.mlp = original_decoder_layer.mlp
        self.input_layernorm = original_decoder_layer.input_layernorm
        self.post_attention_layernorm = original_decoder_layer.post_attention_layernorm

        self.pass_mha = False
        self.pass_mlp = False
        self.input = None
        self.output = None

        self.s = None
        self.bias = None

    def turn_off(self):
        self.pass_mha = True
        self.pass_mlp = True

    def turn_on(self):
        self.pass_mha = False
        self.pass_mlp = False

    def turn_off_mha(self):
        self.pass_mha = True

    def turn_on_mha(self):
        self.pass_mha = False

    def turn_off_mlp(self):
        self.pass_mlp = True

    def turn_on_mlp(self):
        self.pass_mlp = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if not self.pass_mha:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual.to(hidden_states.device) + hidden_states
        else:
            self_attn_weights = None

        if not self.pass_mlp:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual.to(hidden_states.device) + hidden_states

        if self.bias is None:
            outputs = (hidden_states,)
        else:
            outputs = (hidden_states * self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device), )

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


def block_replace(model):
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        model.model.layers[i] = OnOff_LlamaDecoderLayer(model.model.layers[i])
    print("Replacement complete.")

    return model


def turn_off_layer(model, layer_idx):
    model.model.layers[layer_idx].turn_off()


def turn_on_layer(model, layer_idx):
    model.model.layers[layer_idx].turn_on()


def turn_off_mha(model, layer_idx):
    model.model.layers[layer_idx].turn_off_mha()


def turn_on_mha(model, layer_idx):
    model.model.layers[layer_idx].turn_on_mha()


def turn_off_mlp(model, layer_idx):
    model.model.layers[layer_idx].turn_off_mlp()


def turn_on_mlp(model, layer_idx):
    model.model.layers[layer_idx].turn_on_mlp()


def scan(model, num_blocks):
    alive_list = []
    skip_list = []

    for i in range(num_blocks):
        if model.model.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.layers[i].pass_layer == False:
            alive_list.append(i)

    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
    )


class DynamicLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, pruning_num, base_model=None, tokenizer=None, router_model=None, router_tokenizer=None, params_dict=None):

        super().__init__(config)
        self.n_layers = config.num_hidden_layers

        self.tokenizer = tokenizer
        self.router_tokenizer = router_tokenizer

        if base_model is not None:
            self.model = base_model
        else:
            print('no llm model loaded...')
        if router_model is not None:
            self.router = router_model
        else:
            print('no trained router loaded...')
        self.router.eval()

        self.seqlen = 2048
        self.params_dict = params_dict
        self.pruning_num = pruning_num

        self.input_set = {}

    def get_skip_mask(self, router_logits):

        probabilities = router_logits
        top_indices = torch.argmin(probabilities, dim=-1)
        predicted_label = top_indices.item()  # (1,1) -> int

        skip_layer = [idx for idx in range(predicted_label, predicted_label + self.pruning_num)]
        self.base_model.model.layers[predicted_label].s = self.params_dict[predicted_label][0]
        self.base_model.model.layers[predicted_label].bias = self.params_dict[predicted_label][1]

        return skip_layer

    def forward(self, input_ids=None, attention_mask=None, skip_layer=None, **kwargs):
        seq_len = input_ids.size(1)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((1, seq_len), device=device)

        if skip_layer is None:
            with torch.no_grad():
                input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

                answer_index = input_text.find("Answer")

                if answer_index != -1:
                    question_input = input_text[:answer_index]
                else:
                    match = re.search(r":\s*([^\.]*\.)", input_text)
                    if match:
                        question_input = match.group(1).strip()
                    else:
                        words = input_text.split()
                        question_input = " ".join(words[:7])

                if question_input in self.input_set:
                    skip_layer = self.input_set[question_input]

                else:
                    bert_inputs = self.bert_tokenizer(
                        input_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)

                    router_outputs = self.router(
                        input_ids=bert_inputs['input_ids'],
                        attention_mask=bert_inputs['attention_mask']
                    )
                    router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)

                    skip_layer = self.get_skip_mask(router_logits)
                    self.input_set[question_input] = skip_layer

                # input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                # bert_inputs = self.router_tokenizer(
                #     input_text,
                #     return_tensors='pt',
                #     padding=True,
                #     truncation=True,
                #     max_length=512
                # ).to(device)
                #
                # router_outputs = self.router(
                #     input_ids=bert_inputs['input_ids'],
                #     attention_mask=bert_inputs['attention_mask']
                # )
                # router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)
                #
                # skip_layer = self.get_skip_mask(router_logits)

        for idx in skip_layer:
            turn_off_layer(self.model, idx)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for idx in skip_layer:
            turn_on_layer(self.model, idx)
        self.base_model.model.layers[skip_layer[0]].s = None
        self.base_model.model.layers[skip_layer[0]].bias = None

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None
        )

