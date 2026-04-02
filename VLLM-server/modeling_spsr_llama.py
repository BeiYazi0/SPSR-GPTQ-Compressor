import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
from typing import Dict, Optional, Tuple, Union
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.config import ModelConfig

class IdentityNormLike(nn.Module):
    def __init__(self, s: torch.Tensor, bias: torch.Tensor, relu: bool = False):
        super().__init__()
        self.s = nn.Parameter(s)
        self.bias = nn.Parameter(bias)
        self.relu = relu

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if residual is None: # 正常前向
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else: # norm 内部 h + r 作为输入，得到 h， 且 r = h + r
        #     hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        # # 成功将 attn 模块的输出拆分为 h 和 r
        # # Fully Connected 
        # hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states * self.s.to(hidden_states.device) + self.bias.to(hidden_states.device)    
        if residual is not None:
            residual = residual * self.s.to(residual.device) + self.bias.to(residual.device)
            
        return hidden_states, residual

class SPSRLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # 设置基础路径和SPSR配置
        hf_config = vllm_config.model_config.hf_config
        self.base_model_path = hf_config.spsr_metadata["original_model"]
        self.spsr_path = vllm_config.model_config.model

        # __import__('fpdb').ForkedPdb().set_trace()

        if self.base_model_path is None:
            raise ValueError("SPSR model requires `base_model_path` in ModelConfig")

        print(f"Loading base model from: {self.base_model_path}")
        setattr(vllm_config.model_config, "model", self.base_model_path)

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """重写加载权重方法，先加载基础权重，再应用SPSR"""
        print("Loading base model weights into SPSR model...")
        # __import__('fpdb').ForkedPdb().set_trace()
        super().load_weights(weights)

        if self.spsr_path is not None:
            print(f"Applying SPSR layers from: {self.spsr_path}")
            self.apply_spsr_layers(self.spsr_path)
    
    def apply_spsr_layers(self, spsr_path: str):
        """应用SPSR层替换 - 优化版本"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # 加载轻量的IdentityNormLike层信息
        save_data = torch.load(spsr_path + "/spsr_layers.pth", map_location=device)
        identity_layers_info = save_data['identity_layers']
        
        # 获取层列表
        layers = self.model.layers
        
        # 从后往前替换
        for layer_info in reversed(identity_layers_info):
            start_index = layer_info['start_index']
            end_index = layer_info['end_index']
            s = layer_info['s'].to(device, dtype)
            bias = layer_info['bias'].to(device, dtype)
            relu = layer_info['relu']
            
            identity_layer = IdentityNormLike(s, bias, relu=relu)
            
            # 替换层
            new_layers_list = list(layers[:start_index]) + [identity_layer] + list(layers[end_index:])
            self.model.layers = nn.ModuleList(new_layers_list)
            layers = self.model.layers
            
            print(f"Replaced layers [{start_index}:{end_index}] with IdentityNormLike")
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ SPSR layers applied successfully!")


