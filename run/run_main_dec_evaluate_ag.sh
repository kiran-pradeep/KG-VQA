# source .env
data_mode="dev" # Options: 'dev', 'train' 
data_path="data/MuSiQue/updated_dev.jsonl" # UPDATE THIS WITH THE PATH TO THE TARGET DATASET
# data_path="./data/dev/dev.json" # UPDATE THIS WITH THE PATH TO THE TARGET DATASET

config="./run/configs/DEC_EVAL_BASE_AG/DEC_EVAL_BASE_AG_full.yaml"
# oracle_path="./data/dev/subsampled_schema_refiner_schema.json_llama_3_1_8B.json"
# oracle_path="./data/dev/schema_refiner_schema_llama_3_1_8B.json"
# oracle_path="./data/dev/schema_refiner_schema_llama_3_1_8B.json"

num_workers=8 # Number of workers to use for parallel processing, set to 1 for no parallel processing

# schema pruning options: 'oracle_dev_custom_sc', 'oracle_dev_custom_sr', 'oracle_dev',
# 'oracle_subdev_custom_sc', 'oracle_subdev_custom_sr', 'oracle_subdev',  'complete', 'tentative'


python3 -u ./src/main.py --data_mode ${data_mode} --data_path ${data_path} --config "$config" --start 0 \
        --num_workers ${num_workers} --pick_final_sql true
