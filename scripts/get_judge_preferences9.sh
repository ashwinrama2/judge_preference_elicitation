#!/bin/bash

# base folder
chatbot_root="/global/cfs/cdirs/nstaff/chatbot"
# local folders
script_dir=$(dirname "$(realpath "$0")")
local_code_folder="$(dirname "$script_dir")/evaluate"
# main folders
models_folder="$chatbot_root/models"
# data folders
python_instance="shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3"

judge_name="Vicuna"

$python_instance $local_code_folder/get_judge_prefs9.py --models_folder $models_folder --judge_name $judge_name --curr_dir $local_code_folder
