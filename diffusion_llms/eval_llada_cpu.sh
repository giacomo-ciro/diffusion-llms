#!/bin/bash
# filepath: /Users/davidebeltrame/Documents/repo/diffusion-llms/diffusion_llms/eval_llada_cpu.sh

# Set the environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# For Mac, we need to specify CPU explicitly for evaluation
# Add these environment variables to force CPU usage
export CUDA_VISIBLE_DEVICES=""
export USE_CUDA=0

# We'll use a smaller batch size and fewer Monte Carlo iterations for CPU evaluation
echo "Running evaluations on CPU. This will be slower than GPU evaluation."

# For Mac/CPU, we'll run a limited test with smaller parameters
python eval_llada.py \
  --tasks hellaswag \
  --num_fewshot 0 \
  --model llada_dist \
  --batch_size 2 \
  --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.5,is_check_greedy=False,mc_num=16,device=\cpu\

# Add more tasks as needed, but keep batch_size and mc_num small for CPU
# python eval_llada.py \
#   --tasks gsm8k \
#   --model llada_dist \
#   --batch_size 1 \
#   --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=256,steps=128,block_length=256,device=\"cpu\"
