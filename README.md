# ParallelKernelBench (PKB)

A benchmark evaluation framework heavily inspired by the Scaling Intelligence Lab's KernelBench project designed to assess the performance of multi-GPU AI workloads. Please contribute!

## Evaluation Framework

Similar to KernelBench, there is a **PyTorch reference** implementing a solution to a particular problem. Note that we are *not assessing the LLM's capability of rewriting NCCL*, rather we are testing how *good* the LLM is at *utilizing NCCL* in creative and performant ways.

If you're familiar with GPU communication, you know that NVIDIA offers two libraries: (1) NCCL and (2) NVSHMEM. NCCL works by essentially putting custom communication kernels on the stream, while NVSHMEM allows you to perform **kernel-initiated** communication which can reduce latency. 

So, the LLM can either choose to make a custom CUDA kernel with NVSHMEM APIs within, or it can call NCCL APIs via PyTorch and put those communication operations on stream. **We leave this as a decision to the LLM, although a human would be useful for guidance!**


NOTE: There is a new feature in NCCL 2.28 called "GIN" that lets you do put/get operations directly from GPU kernels!

Put in very simple terms:
```
GOAL: "Given this problem and this GPU topology, write the FASTEST possible solution."

The LLM can choose:
✅ Pure PyTorch + torch.distributed
✅ PyTorch + custom CUDA kernels (with PyTorch for comm)
✅ Pure CUDA + NCCL calls
✅ Pure CUDA + NVSHMEM
✅ Any hybrid approach
```

## Topology-Awareness
We can piggyback off of NCCL's topology detection, which outputs an XML of the hardware topology. We just need to set this environment variable:
```
export NCCL_TOPO_DUMP_FILE=system_topology.xml
```
In order to get the XML file, which we can then parse into a more LLM-compatible JSON file using the provided python script (parse_nccl_topo.py)


## Running instructions
For now you can run the reference implementation using:
```bash
sbatch tests/run_reference.sh
```
This will save the resulting tensors on each rank to the logs folder.

### Triton Distributed Instructions
```bash
pip install https://github.com/ByteDance-Seed/Triton-distributed/releases/download/v0.0.1-rc/triton_dist-3.4.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```


## Problems

### Level 1: PyTorch + torch.distributed
* TODO

### Level 2: Custom CUDA + PyTorch Communication
* TODO

### Level 3: CUDA + NVSHMEM (Kernel-Initiated Communication)
* TODO


## Prompt Template Structure

```
=== PROBLEM STATEMENT ===
{High-level description of what needs to be parallelized}

=== TASK ===
Implement an efficient multi-GPU solution for {problem_name}.

Your implementation will run across {num_ranks} processes (ranks), one per GPU.
You may use any combination of:
- PyTorch and torch.distributed
- Custom CUDA kernels
- NCCL (via torch.distributed or direct calls)
- NVSHMEM

Your goal is to maximize performance. Choose the right tool for the problem.

=== FUNCTION SIGNATURE ===
def solution(input_tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Args:
        input_tensor: Input data on this rank. Shape: {input_shape}, dtype: {dtype}
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes ({num_gpus})
    
    Returns:
        output_tensor: Result on this rank. Shape: {output_shape}, dtype: {dtype}
    
    Notes:
        - Each rank runs on GPU {rank}
        - torch.distributed is already initialized with NCCL backend
        - All ranks must participate in collective operations
    """
    pass

=== INPUT/OUTPUT SPECIFICATION ===
Input per rank: {detailed_input_spec}
Output per rank: {detailed_output_spec}

Example:
  Rank 0 receives: tensor of shape [2, 1024] with values [[1, 2, ...], [3, 4, ...]]
  Rank 1 receives: tensor of shape [2, 1024] with values [[5, 6, ...], [7, 8, ...]]
  
  Rank 0 should output: {expected_output_rank_0}
  Rank 1 should output: {expected_output_rank_1}

=== REFERENCE IMPLEMENTATION ===
{ref_impl_src} 

# Single-GPU reference implementation (naive)
# Your multi-GPU version should match this output but run faster

=== SYSTEM TOPOLOGY ===
{
  "num_nodes": 1,
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA A100-SXM4-40GB",
      "compute_capability": "8.0",
      "memory_gb": 40,
      "memory_bandwidth_gb_s": 1555,
      "connected_to": [1, 2],
      "nvlink_peers": [1],
      "nvlink_bandwidth_gb_s": 300,
      "pcie_gen": 4,
      "pcie_lanes": 16,
      "pcie_bandwidth_gb_s": 32,
      "pci_bus_id": "0000:01:00.0",
      "numa_node": 0,
      "cpu_affinity": [0]
    },
    {
      "id": 1,
      "name": "NVIDIA A100-SXM4-40GB",
      "compute_capability": "8.0",
      "memory_gb": 40,
      "memory_bandwidth_gb_s": 1555,
      "connected_to": [0, 2],
      "nvlink_peers": [0],
      "nvlink_bandwidth_gb_s": 300,
      "pcie_gen": 4,
      "pcie_lanes": 16,
      "pcie_bandwidth_gb_s": 32,
      "pci_bus_id": "0000:02:00.0",
      "numa_node": 0,
      "cpu_affinity": [0]
    },
    {
      "id": 2,
      "name": "NVIDIA A100-SXM4-40GB",
      "compute_capability": "8.0",
      "memory_gb": 40,
      "memory_bandwidth_gb_s": 1555,
      "connected_to": [0, 1],
      "nvlink_peers": [],
      "nvlink_bandwidth_gb_s": 0,
      "pcie_gen": 4,
      "pcie_lanes": 16,
      "pcie_bandwidth_gb_s": 32,
      "pci_bus_id": "0000:03:00.0",
      "numa_node": 1,
      "cpu_affinity": [1]
    }
  ],
  "topology_notes": "GPUs 0 and 1 are connected via NVLink (600 GB/s bidirectional). GPU 2 is connected via PCIe Gen4 x16 and is on a different NUMA node."
}

=== ENVIRONMENT ===
- PyTorch {version} with CUDA {cuda_version}
- NCCL {nccl_version}
- NVSHMEM {nvshmem_version} (if using Level 3)
- Available libraries: numpy, cupy, torch.distributed

=== CONSTRAINTS ===
- All ranks must produce correct outputs
- Solution must handle the provided input shapes
- Memory usage must fit within GPU memory limits
- No host-device transfers except for initial input and final output

=== EVALUATION METRICS ===
Primary: End-to-end latency (ms)
Secondary: 
  - Throughput (samples/sec or GB/s)
  - Memory efficiency (peak memory usage)
  - Communication efficiency (time in communication vs computation)

Baseline performance: {baseline_latency} ms
Target: < {target_latency} ms

=== TEST CASES ===
Test 1 (Correctness):
  Input: {small_test_input}
  Expected Output: {small_test_output}

Test 2 (Performance):
  Input: {large_test_input}
  Performance target: < {target_latency} ms
```





## New Insights and Developments
- Kudos to Stuart Sul for the following footnotes:
  - Fundamental tradeoff between performance and simplicity: NCCL is the LEAST efficient. 
  - For intranode, you don't need a framework like NVSHMEM: everything is handled with UVA under the hood (plain memory access) on newer GPUs.
  - This only comes into play when you're doing internode (which we should be trying to do)
  - NVSHMEM kernels aren't that intimidating if you know what you're doing: comms is shorter than compute and nvshmem is relatively straightforward.
- GPU Mode insights:
  - NVSHMEM has a lot of frameworks built on top of it, lets you do kernel-initiated communication
  - NVSHMEM4Py and NCCL4Py are also valid frameworks! Worth looking into.
- AstraSim seems like an interesting framework for simulating COMPUTE and COMM phases for different topologies: but it can't run actual kernels which is what we want to do. Maybe we write one kernel, analyze compute/comm phases, then input to astrasim for different topologies???