import torch
import rattle_hard_cuda
import time
from rattle_step import Manifold


def random_point_on_sphere(bs,dim):
    x = torch.randn(bs,dim, device='cuda', dtype=torch.float32)
    x = x / x.norm(p=2,dim=-1, keepdim=True)
    return x, torch.ones_like(x)

batch_size=1
# Example usage (assuming you adapted the interface)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For all GPUs
x, v =  random_point_on_sphere(batch_size,3)

h = 0.000001
t1 = time.time()
rattle_hard_cuda.rattle_hard(x, v, h, 1000)
t2 = time.time()
print(f'time per iteration = {t2-t1} ms')

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For all GPUs
x, v =  random_point_on_sphere(batch_size,3)

manifold = Manifold([lambda x : (x[:,0]**2 + x[:,1]**2 + x[:,2]**2)-1]) # creating a spherical manifold

t1 = time.time()
for _ in range(1000):
    x, v = manifold.rattle_step(x,v,h)
t2 = time.time()
print(f'time per iteration = {t2-t1} ms')



