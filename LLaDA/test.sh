python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:1" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv" \
    --safety

python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:1" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv" \
    --safety

# /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:2" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv" \
    --safety
/opt/tiger/sft_entity/LLaDA/generation_alpha.py

python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda:4" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv"

# /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:5" \
    --input_path "/opt/tiger/sft_entity/dllm-safety/gcg.csv"

# --remasking adaptive_step --alpha0 0.3
python generate_detection.py \
  --model_name "LLaDA-8B-Instruct" \
  --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
  --input_path /opt/tiger/sft_entity/dllm-safety/dija_advbench.json \
  --input_format json \
  --json_field "refined prompt" \
  --fill_all_masks \
  --steps 64 \
  --gen_length 128 \
  --block_length 128 \
  --temperature 0.5 \
  --device cuda \
  --sp_mode hidden \
  --debug_print
#   --remasking adaptive_step --alpha0 0.4
