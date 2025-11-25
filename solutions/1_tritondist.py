import torch
import torch.distributed as dist
import triton
import triton.language as tl

@triton.jit
def identity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)

def solution(tensor):
    """
    Example solution using Triton for local computation and torch.dist for communication.
    This matches the reference behavior (global sum) but runs a Triton kernel locally first.
    """
    assert dist.is_initialized()
    
    # 1. Allocate output
    out = torch.empty_like(tensor)
    
    # 2. Run Triton kernel (local copy/identity) to demonstrate Triton integration
    n_elements = tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    identity_kernel[grid](tensor, out, n_elements, BLOCK_SIZE=1024)
    
    # 3. Perform All-Reduce to match the reference semantics (sum across ranks)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    
    return out
