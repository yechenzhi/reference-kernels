from task import input_t, output_t
import torch
import torch.nn.functional as F

class DisableCuDNNTF32:
    def __init__(self):
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        pass

    def __enter__(self):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic

def custom_kernel(data: input_t) -> output_t:
    with DisableCuDNNTF32():
        input_tensor, kernel = data
        return F.conv2d(
            input_tensor,
            kernel,
            stride=1,
            padding=0
        )
