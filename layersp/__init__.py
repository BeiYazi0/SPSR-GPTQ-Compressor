from .owl import OWL
from .dlp import DLP
from .atp import ATP

class Uniform:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.A = 5

    def get_layer_sp(self, args):
        if "Llama" in args.base_model or "llama" in args.base_model:
            layers = self.model.model.layers
        elif "opt" in args.base_model:
            layers = self.model.model.decoder.layers
        elif "Qwen" in args.base_model:
            layers = self.model.transformer.h
        else:
            layers = self.model.model.layers
        return [args.final_s] * len(layers)
