import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank (torch.cuda.current_device())
                           # All ranks must provide tensors of identical shape and dtype.
                           # The tensor values may differ across ranks.
) -> torch.Tensor:  # Returns: A tensor of shape [numel / world_size] containing this rank's portion of the
                    #          reduce-scatter result. The input tensor is first reduced (summed) across all
                    #          ranks, then the result is scattered so each rank receives an equal chunk.
    """
    NCCL reduce-scatter (sum) operation over all ranks.
    
    Performs a reduce-scatter collective operation that:
    1. Reduces (sums) the input tensor across all ranks
    2. Scatters the result so each rank receives 1/world_size of the reduced tensor
    
    Each rank receives a unique, non-overlapping chunk of the reduced result.
    The chunks are ordered by rank: rank 0 gets the first chunk, rank 1 gets the second, etc.
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensor must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must call this function with tensors of the same shape and dtype
        - Tensor's total number of elements must be divisible by world_size
    
    Returns:
        A new tensor (1D) of shape [numel / world_size] containing this rank's portion
        of the reduce-scatter result (elementwise sum across ranks, then scattered).
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()
    
    # Split along first dimension
    assert tensor.shape[0] % world_size == 0
    chunk_size = tensor.shape[0] // world_size
    output_shape = (chunk_size,) + tensor.shape[1:]
    
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    dist.reduce_scatter_tensor(output, tensor, op=dist.ReduceOp.SUM)
    
    return output