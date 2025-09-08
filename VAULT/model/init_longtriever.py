import json
import os
import torch
import sys
import dotenv
dotenv.load_dotenv()

from modeling_longtriever import Longtriever
from modeling_hierarchical import HierarchicalLongtriever
from arguments import DataTrainingArguments, ModelArguments
from modeling_utils import log_message

from transformers import AutoModel, AutoTokenizer, HfArgumentParser,TrainingArguments

STORAGE_DIR = os.getenv("STORAGE_DIR")

def init_longtriever_params(base_model, longtriever):
    # Get the state_dict of the BERT encoder
    base_state_dict = base_model.state_dict()

    # Get the state_dict of the Longtriever model
    longtriever_state_dict = longtriever.state_dict()

    # Map weights from BERT to Longtrieverss
    new_state_dict = {}
    for bert_key, bert_value in base_state_dict.items():
        # Replace layer names to match Longtriever's naming convention
        if "layer" in bert_key:
            # Example: Map BERT's "layer.X" to Longtriever's "text_encoding_layers.X"
            new_key = bert_key.replace("layer", "text_encoding_layer")
            if new_key in longtriever_state_dict:
                new_state_dict[new_key] = bert_value
            else:
                print(f"Key {new_key} not found in Longtriever state_dict.")

            # Example: Map BERT's "layer.X" to Longtriever's "information_exchanging.X"
            new_key = bert_key.replace("layer", "information_exchanging_layer")
            if new_key in longtriever_state_dict:
                new_state_dict[new_key] = bert_value
            else:
                print(f"Key {new_key} not found in Longtriever state_dict.")
        else:
            if bert_key in longtriever_state_dict:
                new_state_dict[bert_key] = bert_value
            else:
                print(f"Key {bert_key} not found in Longtriever state_dict.")

    # Update Longtriever's state_dict with the new weights
    longtriever_state_dict.update(new_state_dict)

    # Load the updated state_dict into Longtriever
    longtriever.load_state_dict(longtriever_state_dict)

    return longtriever

def check_weights(base_model, longtriever):
    # Compare weights
    lt_weights = longtriever.state_dict().keys()

    for name, param in base_model.named_parameters():
        text_name = name.replace("layer", "text_encoding_layer")
        # print(name)
        if text_name in lt_weights:
            lt_text_param = longtriever.state_dict()[text_name]
            if torch.equal(param, lt_text_param):
                print(f"Layer {text_name} matches")
                pass
            else:
                print(f"Layer {text_name} does not match")
                pass
        else:
            print(f"Layer {text_name} not found.")
            pass

        info_name = name.replace("layer", "information_exchanging_layer")
        if info_name in lt_weights:
            lt_info_param = longtriever.state_dict()[info_name]
            if torch.equal(param, lt_info_param):
                print(f"Layer {info_name} matches")
                pass
            else:
                print(f"Layer {info_name} does not match")
                pass
        else:
            print(f"Layer {info_name} not found.")
            pass

    return

if __name__ == "__main__":
    # Parse all the arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) <=1 :
        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, data_args, training_args = parser.parse_json_file(json_file="/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/longtriever/configs/longtriever_test.json", allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Main arguments
    model_path = training_args.output_dir
    
    # Parameters
    model_path = data_args.base_model
    
    longtriever_dir = os.path.join(STORAGE_DIR, "models", "longtriever_og")
    os.makedirs(longtriever_dir, exist_ok=True)

    pretrained_dir = os.path.join(longtriever_dir, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)

    save_path = os.path.join(pretrained_dir, model_path.split("/")[-1])
    if data_args.base_model_postfix!=None or data_args.base_model_postfix != "false":
        if data_args.base_model_postfix=="true":
            postfix = len(os.listdir(pretrained_dir)) + 1
        else:
            postfix = data_args.base_model_postfix
            save_path = save_path + f"-{postfix}"
            print("base_model_postfix enabled, adding postfix to model name. New save path: ", save_path) 

    if not training_args.overwrite_output_dir and os.path.exists(save_path):
        print(f"Model {save_path} already exists, skipping.")
    else:
        # Load tokenizer, there should be no issues here
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if model_args.model_type=="longtriever":
            model_type = Longtriever
        elif model_args.model_type=="hierarchical":
            model_type = HierarchicalLongtriever

        # Load the base model
        base_model = AutoModel.from_pretrained(model_path)
        # Stick the parameters that match the Longtriever model
        longtriever = model_type.from_pretrained(
                data_args.base_model, 
                ablation_config=model_args.ablation_config, 
                doc_token_init=model_args.doc_token_init
            )

        # Initialize both Longtriever encoders with the base model's encoder
        init_longtriever_params(base_model, longtriever)
        
        # Ensure the parameters are OK
        check_weights(base_model, longtriever)

        # # Save everything
        longtriever.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
