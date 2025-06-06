# import fm
# from peft import get_peft_model, LoraConfig, TaskType

# def get_model():
#     # Base
#     model, alphabet = fm.pretrained.rna_fm_t12()
# # not sure if this is correct. 
#     # LoRA stuff
#     peft_config = LoraConfig(
#         r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias="none",
#         target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
#         inference_mode = False, 
#         task_type=TaskType.FEATURE_EXTRACTION

#     )
#     return get_peft_model(model, peft_config), alphabet


### Custom LORA
import torch
import torch.nn as nn
import fm

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8):
        super().__init__()
        self.original_layer = original_layer  # Preserve original layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = r

        self.lora_A = nn.Parameter(torch.randn(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features, requires_grad=True))

        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False

    @property
    def weight(self):
        """Expose original layer's weight for compatibility"""
        return self.original_layer.weight

    @property
    def bias(self):
        """Expose original layer's bias for compatibility"""
        return self.original_layer.bias

    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B)
    
    
def add_lora(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for layer in model.layers:
        attn = layer.self_attn
        attn.q_proj = LoRALayer(attn.q_proj)
        attn.k_proj = LoRALayer(attn.k_proj)
        attn.v_proj = LoRALayer(attn.v_proj)
        attn.out_proj = LoRALayer(attn.out_proj)
        
    return model
