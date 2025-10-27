pip install -r requirements.txt
export VAULT_DIR=/Tmp/lvpoellhuber/datasets/vault


# bash scripts/process_all.sh

# bash scripts/hierarchical.sh
bash scripts/hierarchical-passage.sh

# bash scripts/bert.sh

# bash scripts/dpr.sh

# bash scripts/longtriever.sh

# python baselines.py

# TODO: Remove and uncomment from process_all.sh
# echo "Processing Doris-MAE." 
# python preprocessing/doris_mae.py

