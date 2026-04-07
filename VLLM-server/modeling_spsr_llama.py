import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import json
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
    

class IdentityLinearLike(nn.Module):
    def __init__(self, s, bias):
        super().__init__()

        self.s = s
        self.bias = bias

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states @ self.s.to(device=hidden_states.device) + self.bias.to(device=hidden_states.device)
        if residual is not None:
            residual = residual @ self.s.to(residual.device) + self.bias.to(residual.device)
     
        return hidden_states, residual


class SPSRLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        # 设置基础路径和SPSR配置
        hf_config = vllm_config.model_config.hf_config
        self.base_model_path = hf_config.spsr_metadata["original_model"]
        self.spsr_path = vllm_config.model_config.model
        self.weight_type = hf_config.spsr_metadata.get("weight_type", "scale")  

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

            # self.apply_lora_weights(self.spsr_config['spsr_path'])

    
    def apply_spsr_layers(self, spsr_path: str):
        """应用SPSR层替换 - 优化版本"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        weight_type = self.weight_type
        
        # 加载轻量的层信息
        save_data = torch.load(os.path.join(spsr_path, "spsr_layers.pth"), map_location=device)
        identity_layers_info = save_data['identity_layers']
        
        # 获取层列表
        layers = self.model.layers
        
        # 从后往前替换
        for layer_info in reversed(identity_layers_info):
            start_index = layer_info['start_index']
            end_index = layer_info['end_index']
            s = layer_info['s'].to(device, dtype)
            bias = layer_info['bias'].to(device, dtype)
            relu = layer_info.get('relu', False)
            
            # 根据 weight_type 创建相应的层
            if weight_type == 'linear':
                identity_layer = IdentityLinearLike(s, bias)
                print(f"Replaced layers [{start_index}:{end_index}] with IdentityLinearLike")
            else:
                identity_layer = IdentityNormLike(s, bias, relu=relu)
                print(f"Replaced layers [{start_index}:{end_index}] with IdentityNormLike")
            
            # 替换层
            new_layers_list = list(layers[:start_index]) + [identity_layer] + list(layers[end_index:])
            self.model.layers = nn.ModuleList(new_layers_list)
            layers = self.model.layers
        
        # # 检测并加载 LoRA
        # adapter_bin_path = os.path.join(spsr_path, "adapter_model.bin")
        # adapter_safetensors_path = os.path.join(spsr_path, "adapter_model.safetensors")
        # if os.path.exists(adapter_bin_path) or os.path.exists(adapter_safetensors_path):
        #     from peft import PeftModel
        #     # 注意：在 vLLM 中，模型已经加载，需要手动合并 LoRA 权重
        #     # 这里假设 LoRA 权重可以直接加载并合并
        #     model = PeftModel.from_pretrained(self.model, spsr_path)
        #     self.model = model.merge_and_unload()
        #     print(f"Loaded and merged LoRA weights from {spsr_path}")
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ SPSR layers applied successfully!")

    def apply_lora_weights(self, spsr_path: str):
        """正确合并LoRA权重到SPSR模型中"""
        # 1. 检查必要的LoRA文件
        adapter_config_path = os.path.join(spsr_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print("Skipping LoRA merge process.")
            return
        
        print(f"\n=== Starting LoRA Weight Merge Process ===")
        print(f"Checking for LoRA files in: {spsr_path}")
        
        
        # 2. 读取adapter配置
        print(f"✅ Loading adapter configuration from {adapter_config_path}")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        print(f"Adapter Configuration:")
        print(f"  Target Modules: {adapter_config.get('target_modules', [])}")
        print(f"  Rank (r): {adapter_config.get('r', 8)}")
        print(f"  Alpha: {adapter_config.get('lora_alpha', 16)}")
        print(f"  Dropout: {adapter_config.get('lora_dropout', 0.05)}")
        
        # 3. 获取target modules
        target_modules = adapter_config.get('target_modules', [])
        
        # 4. 加载LoRA权重
        lora_weights = self._load_lora_weights(spsr_path)
        
        # 5. 准备合并参数
        lora_alpha = adapter_config.get('lora_alpha', 16)
        lora_r = adapter_config.get('r', 8)
        scaling_factor = lora_alpha / lora_r
        print(f"LoRA Scaling Factor: alpha/r = {lora_alpha}/{lora_r} = {scaling_factor:.4f}")
        
        # 6. 获取模型的命名参数
        model_named_params = dict(self.named_parameters())
        model_named_modules = dict(self.named_modules())
        
        # 7. 合并LoRA权重到target modules
        print(f"\n=== Merging LoRA weights into target modules ===")
        successfully_merged = 0
        failed_modules = []
        
        for target_module in target_modules:
            print(f"\nProcessing target module: {target_module}")
            
            # 查找模型中匹配的模块
            matched_modules = []
            for module_name, module in model_named_modules.items():
                if target_module in module_name:
                    matched_modules.append((module_name, module))
            
            if not matched_modules:
                print(f"  ❌ No modules found matching '{target_module}'")
                failed_modules.append(target_module)
                continue
            
            print(f"  ✅ Found {len(matched_modules)} matching modules:")
            for module_name, _ in matched_modules:
                print(f"    - {module_name}")
            
            # 处理每个匹配的模块
            for module_name, module in matched_modules:
                try:
                    if isinstance(module, (nn.Linear, RowParallelLinear, MergedColumnParallelLinear, QKVParallelLinear)):
                        # 处理线性层
                        success = self._merge_lora_into_linear_module(
                            module_name, module, lora_weights, 
                            scaling_factor, model_named_params
                        )
                        if success:
                            successfully_merged += 1
                    else:
                        print(f"  ⚠️  Module {module_name} is not a linear layer, skipping")
                
                except Exception as e:
                    print(f"  ❌ Error merging LoRA into {module_name}: {str(e)}")
                    failed_modules.append(module_name)
        
        # 8. 总结合并结果
        print(f"\n=== LoRA Merge Summary ===")
        print(f"Successfully merged LoRA weights into {successfully_merged} modules")
        if failed_modules:
            print(f"Failed to merge into {len(failed_modules)} modules:")
            for failed in failed_modules:
                print(f"  - {failed}")
        
        # 9. 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ LoRA weight merge completed!")
        return successfully_merged > 0

    def _load_lora_weights(self, spsr_path: str) -> Dict[str, torch.Tensor]:
        """加载LoRA权重文件，支持bin和safetensors格式"""
        lora_weights = {}
        
        # 检查adapter_model.bin
        adapter_bin_path = os.path.join(spsr_path, "adapter_model.bin")
        if os.path.exists(adapter_bin_path):
            print(f"✅ Loading LoRA weights from {adapter_bin_path}")
            lora_weights.update(torch.load(adapter_bin_path, map_location='cpu'))
        
        # 检查adapter_model.safetensors
        adapter_safetensors_path = os.path.join(spsr_path, "adapter_model.safetensors")
        if os.path.exists(adapter_safetensors_path):
            print(f"✅ Loading LoRA weights from {adapter_safetensors_path}")
            
            from safetensors.torch import load_file
            lora_weights.update(load_file(adapter_safetensors_path))
        
        if not lora_weights:
            print("❌ No LoRA weight files found")
            return {}
        
        print(f"✅ Loaded {len(lora_weights)} LoRA weight tensors")
        
        # 清理键名（移除可能的前缀）
        cleaned_weights = {}
        for key, value in lora_weights.items():
            # 移除常见前缀
            clean_key = key.replace('base_model.model.', '')
            clean_key = clean_key.replace('model.', '')
            cleaned_weights[clean_key] = value
        
        return cleaned_weights

    def _merge_lora_into_linear_module(self, module_name: str, module: nn.Module, 
                                      lora_weights: Dict[str, torch.Tensor], 
                                      scaling_factor: float, 
                                      model_named_params: Dict[str, nn.Parameter]) -> bool:
        """将LoRA权重合并到指定的线性模块中"""
        print(f"  🔄 Merging LoRA into: {module_name}")
        
        # 1. 确定基础权重参数名
        base_weight_name = f"{module_name}.weight"
        base_bias_name = f"{module_name}.bias"
        
        print(f"    Base weight name: {base_weight_name}")
        
        # 2. 检查基础权重是否存在
        if base_weight_name not in model_named_params:
            print(f"    ❌ Base weight not found: {base_weight_name}")
            return False
        
        base_weight_param = model_named_params[base_weight_name]
        base_weight = base_weight_param.data
        device = base_weight.device
        dtype = base_weight.dtype
        
        print(f"    Base weight shape: {base_weight.shape}, dtype: {dtype}, device: {device}")
        
        # 3. 查找对应的LoRA权重
        lora_A_name = None
        lora_B_name = None
        
        # 尝试不同的命名模式
        possible_patterns = [
            f"{module_name}.lora_A.weight",
            f"{module_name}.lora_A.default.weight",
            f"{module_name}.lora_B.weight",
            f"{module_name}.lora_B.default.weight",
            f"{module_name}.lora_embedding_A",
            f"{module_name}.lora_embedding_B"
        ]
        
        for pattern in possible_patterns:
            if pattern in lora_weights:
                if 'lora_A' in pattern or 'lora_embedding_A' in pattern:
                    lora_A_name = pattern
                elif 'lora_B' in pattern or 'lora_embedding_B' in pattern:
                    lora_B_name = pattern
        
        if lora_A_name is None or lora_B_name is None:
            print(f"    ❌ Could not find LoRA weights for {module_name}")
            print(f"    Available LoRA keys containing '{module_name}':")
            for key in lora_weights.keys():
                if module_name in key:
                    print(f"      - {key}")
            return False
        
        print(f"    ✅ Found LoRA weights:")
        print(f"      lora_A: {lora_A_name}")
        print(f"      lora_B: {lora_B_name}")
        
        # 4. 加载LoRA权重
        lora_A = lora_weights[lora_A_name].to(device, dtype)
        lora_B = lora_weights[lora_B_name].to(device, dtype)
        
        print(f"    lora_A shape: {lora_A.shape}, lora_B shape: {lora_B.shape}")
        
        # 5. 验证维度兼容性
        if lora_A.dim() > 2 or lora_B.dim() > 2:
            print(f"    ❌ LoRA weights have unexpected dimensions (A: {lora_A.dim()}, B: {lora_B.dim()})")
            return False
        
        # 处理嵌入层特殊情况
        is_embedding = 'embed_tokens' in module_name or 'lm_head' in module_name
        if is_embedding:
            print(f"    ⚠️  This is an embedding layer - special handling may be needed")
            # 嵌入层的LoRA处理通常不同，这里简单跳过
            print(f"    ⚠️  Skipping embedding layer LoRA merge for {module_name}")
            return False
        
        # 6. 计算LoRA增量
        try:
            if lora_A.dim() == 1 and lora_B.dim() == 1:
                # 1D情况（不太常见）
                lora_delta = torch.outer(lora_B, lora_A) * scaling_factor
            else:
                # 标准2D情况
                lora_delta = (lora_B @ lora_A) * scaling_factor
            
            print(f"    LoRA delta shape: {lora_delta.shape}")
            
            # 7. 验证维度匹配
            if lora_delta.shape != base_weight.shape:
                print(f"    ❌ Shape mismatch: LoRA delta {lora_delta.shape} vs base weight {base_weight.shape}")
                
                # 尝试转置
                if lora_delta.T.shape == base_weight.shape:
                    print(f"    ⚠️  Transposing LoRA delta to match base weight shape")
                    lora_delta = lora_delta.T
                else:
                    # 尝试其他维度匹配
                    if lora_delta.shape[0] == base_weight.shape[1] and lora_delta.shape[1] == base_weight.shape[0]:
                        print(f"    ⚠️  Transposing LoRA delta (dimensions swapped)")
                        lora_delta = lora_delta.T
                    else:
                        print(f"    ❌ Cannot resolve shape mismatch")
                        return False
            
            # 8. 合并权重
            with torch.no_grad():
                merged_weight = base_weight + lora_delta
                base_weight_param.data.copy_(merged_weight)
            
            print(f"    ✅ Successfully merged LoRA weights into {module_name}")
            return True
            
        except Exception as e:
            print(f"    ❌ Error during LoRA merge: {str(e)}")
            return False


