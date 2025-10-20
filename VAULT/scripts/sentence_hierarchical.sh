cd dev
global_model_path=$STORAGE_DIR'/models/vault/hotpotqa_sentence'

# # Hardcoded to be HotPotQA currently -> change later for other datasets
# python preprocess.py


task="hotpotqa_sentence"

exp_name=$task'-dpr'
dpr_model_path=$global_model_path'/dpr'
echo $dpr_model_path
if [[ ! -d $dpr_model_path ]]; then
  mkdir -p $dpr_model_path
fi
export EXP_NAME=$exp_name

# python index_evaluate.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/vault/$task/corpus.db \
#         --qrels_file $STORAGE_DIR/datasets/vault/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/vault/$task/queries.jsonl \
#         --streaming True \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type dpr\
#         --ctx_model_name_or_path facebook/dpr-ctx_encoder-multiset-base \
#         --q_model_name_or_path facebook/dpr-question_encoder-multiset-base \
#         --evaluate False \
#         --output_dir $dpr_model_path \
#         --do_train False \
#         --num_train_epochs 6 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir False \
#         --dataloader_num_workers 16 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --segments true \



task="hotpotqa"

exp_name=$task'_sentence-hierarchical'
hier_model_path=$global_model_path'/hierarchical'
echo $hier_model_path
if [[ ! -d $hier_model_path ]]; then
  mkdir -p $hier_model_path
fi
export EXP_NAME=$exp_name

echo Training on documents.
torchrun --nproc_per_node=1 train.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/vault/wikipedia/corpus.db \
        --qrels_file $STORAGE_DIR/datasets/vault/$task/qrels/train.tsv \
        --query_file $STORAGE_DIR/datasets/vault/$task/queries.jsonl \
        --streaming True \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical \
        --index_dir $dpr_model_path \
        --output_dir $hier_model_path \
        --do_train True \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 16 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --segments true \


echo Evaluating.

python index_evaluate.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/vault/wikipedia/corpus.db \
        --qrels_file $STORAGE_DIR/datasets/vault/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/vault/$task/queries.jsonl \
        --streaming True \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical\
        --model_name_or_path $hier_model_path \
        --index_dir $dpr_model_path \
        --evaluate True \
        --output_dir $hier_model_path \
        --do_train False \
        --num_train_epochs 6 \
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
        --segments true \
