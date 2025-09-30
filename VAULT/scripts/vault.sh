pip install -r requirements.txt

export VAULT_DIR=/Tmp/lvpoellhuber/datasets/vault

bash scripts/process_all.sh

bash scripts/vault_experiments.sh

# bash scripts/bert.sh

bash scripts/dpr.sh

# TODO: Remove and uncomment from process_all.sh
echo "Processing Doris-MAE." 
python preprocessing/doris_mae.py

python baselines.py

# If everything crashes, to allow me time to fix it
sleep 3h