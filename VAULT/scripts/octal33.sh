export VAULT_DIR=/Tmp/lvpoellhuber/datasets/vault

bash scripts/hierarchical.sh

bash scripts/bert.sh

bash scripts/dpr.sh

# Longtriever

task="trec-covid"

exp_name=$task'-longtriever'
model_path=$STORAGE_DIR'/models/vault/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
export EXP_NAME=$exp_name

echo Training on documents.
torchrun --nproc_per_node=1 train.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/vault/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/vault/$task/qrels/train.tsv \
        --query_file $STORAGE_DIR/datasets/vault/$task/queries.jsonl \
        --streaming False \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type longtriever \
        --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
        --output_dir $model_path \
        --do_train True \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --per_device_train_batch_size 6 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 16 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":false, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'


echo Evaluating.

python evaluate.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/vault/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/vault/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/vault/$task/queries.jsonl \
        --streaming False \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type longtriever\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir False \
        --dataloader_num_workers 16 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":false, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'


python preprocessing/msmarco.py