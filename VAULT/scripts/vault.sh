pip install -r requirements.txt

export VAULT_DIR=/Tmp/lvpoellhuber/datasets/vault

bash scripts/bert.sh

bash scripts/vault_experiments.sh

bash scripts/process_all.sh

python baselines.py
