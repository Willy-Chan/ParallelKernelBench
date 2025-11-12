import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank (torch.cuda.current_device())
                           # All ranks must provide tensors of identical shape and dtype.
                           # Each rank typically provides different tensor values (unique per-rank data).
) -> torch.Tensor:  # Returns: A tensor containing the concatenation of all ranks' input tensors.
                    #          Shape: [world_size * numel] (flattened concatenation)
                    #          All ranks receive identical output tensors.
    """
    NCCL all-gather operation over all ranks.
    
    Performs an all-gather collective operation that gathers tensors from all ranks
    and concatenates them, with all ranks receiving the complete gathered result.
    
    The output tensor contains chunks ordered by rank: 
    [rank0's tensor | rank1's tensor | rank2's tensor | ... | rankN's tensor]
    
    All ranks receive identical output containing everyone's data.
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensor must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must call this function with tensors of the same shape and dtype
    
    Returns:
        A new tensor (1D) of shape [world_size * numel] where the output contains
        the concatenation of all ranks' input tensors in rank order.
        All ranks receive identical copies of this gathered result.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    world_size = dist.get_world_size()
    
    # Gather along first dimension
    output_shape = (world_size * tensor.shape[0],) + tensor.shape[1:]
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    dist.all_gather_into_tensor(output, tensor)
    
    return output