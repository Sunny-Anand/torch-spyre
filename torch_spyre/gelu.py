import torch
import torch_spyre

import torch.nn.functional as F

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

x = torch.rand(512, 1024, dtype=torch.float16)
x_device = x.to(DEVICE)
compiled = torch.compile(lambda a: F.gelu(a))
compiled_result = compiled(x_device).cpu()
cpu_ref = F.gelu(x)

diff = (compiled_result - cpu_ref).abs().max().item()
print("Compiled GELU ran successfully!")
print("Input shape:", x_device.shape, "Device:", x_device.device)
print("Output shape:", compiled_result.shape)
print("Max absolute difference vs PyTorch CPU reference:", diff)
torch.testing.assert_close(compiled_result, cpu_ref, atol=5e-3, rtol=1e-3)
print("Verification PASSED!")