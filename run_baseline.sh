## LET'S FOCUS ON INTRANODE FOR NOW:

# PART 1: CONSTRUCTING THE PROMPT
#       {BASE PROMPT}
#   Description of the problem/what you want it to do (task and problem statement)
#   Expected input and output, with an example, with datatypes.
#   Function Signature to fill in
#   HOW: Implementation mode (for now just writing a big NVSHMEM kernel)
#   Reference implemnentation (torch distributed)
#   SYSTEM TOPOLOGY: extracted from NCCL automatically





# single node - 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
train.py



### RUNNING ON 2 NODES:
# # 2 nodes
# # on node 0:
# MASTER_ADDR=node0.example.com MASTER_PORT=29500 NODE_RANK=0 WORLD_SIZE=16 \
# torchrun --nnodes=2 --nproc-per-node=8 \
# train.py
# # on node 1:
# MASTER_ADDR=node0.example.com MASTER_PORT=29500 NODE_RANK=1 WORLD_SIZE=16 \
# torchrun --nnodes=2 --nproc-per-node=8 \
# train.py --arg1 ... --argN ...

# # but if you're using slurm then it would look like this:
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# export MASTER_PORT=29500
# export NODE_RANK=$SLURM_NODEID
# torchrun --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_ON_NODE \
#   train.py --arg1 ... --argN ...



  # want to auto-extract the system's topology with:
  export NCCL_TOPO_DUMP_FILE=system_topology.xml  # -> parse into JSON or something more managable!