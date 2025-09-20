python /opt/tiger/sft_entity/dllm-safety/Dream/dream_batch_infer.py \
  --model_path "Dream-org/Dream-v0-Instruct-7B" \
  --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
  --steps 64 \
  --gen_length 128 \
  --block_length 128 \
  --temperature 0.5 \
  --top_p 0.95 \
  --alg entropy \
  --alg_temp 0.0 \
  --flush_every 5 \
  --output_dir "./outputs" \
  --output_prefix "result_Dream7B"
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
# "/opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv"

python /opt/tiger/sft_entity/dllm-safety/Dream/dream_batch_infer.py \
  --model_path "Dream-org/Dream-v0-Instruct-7B" \
  --input_path "/opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv" \
  --gen_length 128 --block_length 128 --steps 64 \
  --temperature 0.5 --top_p 0.95 \
  --enable_remask \
  --remask_mode adaptive_step --alpha0 0.3
#   --safety \
# 若需要：--mask_token_id 151666