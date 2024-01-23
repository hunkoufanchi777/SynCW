import torch
import torch.nn as nn
class OutputRecorder:
    def __init__(self):
        self.outputs = []
    def hook_fn(self, module, input, output):
        self.outputs.append(output)
    def register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.register_forward_hook(self.hook_fn)
    def clear_list(self):
        self.outputs.clear() # 清空列表

