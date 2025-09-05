python generate_new.py \
    --model_name "LLaDA-8B-Instruct" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-8B-Instruct" \
    --device "cuda" \
    --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
    --safety

# /opt/tiger/sft_entity/datasets/JailbreakBench__JBB-Behaviors/data/judge-comparison.csv
# /opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv
python generate_new.py \
    --model_name "LLaDA-1.5B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/GSAI-ML__LLaDA-1.5" \
    --device "cuda:1" \
    --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
    --safety

python generate_new.py \
    --model_name "MMaDA-8B" \
    --custom_cache_dir "/opt/tiger/sft_entity/models/Gen-Verse__MMaDA-8B-MixCoT" \
    --device "cuda:3" \
    --input_path "/opt/tiger/sft_entity/datasets/allenai__wildjailbreak/eval/eval_harmful.tsv" \
    --safety
