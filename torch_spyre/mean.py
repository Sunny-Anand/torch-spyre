import torch
import torch_spyre

x = torch.rand(128, 64, dtype=torch.float16)

cpu_result = torch.mean(x, dim=0)
DEVICE = torch.device("spyre") 

x_device = x.to(DEVICE)

compiled = torch.compile(lambda a: torch.mean(a, dim=0))
compiled_result = compiled(x_device).cpu()

diff = (compiled_result - cpu_result).abs().max().item()
print("Compiled Mean (Nonstick Reduction) Ran Successfully!")
print("Input shape:", x_device.shape, "Device:", x_device.device)
print("Output shape:", compiled_result.shape)
print("Max absolute difference vs PyTorch CPU reference:", diff)
torch.testing.assert_close(compiled_result, cpu_result, atol=5e-3, rtol=1e-3)
print("Verification PASSED!")