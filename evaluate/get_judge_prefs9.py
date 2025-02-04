import sys
import csv
import os
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')));
import argparse
import asyncio
import json
import inspect
import lmntfy.models.llm as module
from pathlib import Path
from random import shuffle


def parse_args():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_folder", default="../models",type=Path, help="path to the folder containing all the models")
    parser.add_argument("--judge_name", default="Judge", type=str, help="LLM of the judge evaluator")
    parser.add_argument("--curr_dir", default=".", type=Path, help="path to the folder containing test questions")

    args = parser.parse_args()
    return args


async def main():
    # process command line arguments
    args = parse_args()
    models_folder = args.models_folder
    judge_name = args.judge_name
    curr_dir = args.curr_dir
    
    #loads judge information
    judges_dict = {name:obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and issubclass(obj, module.LanguageModel) and obj is not module.LanguageModel}
    if judge_name not in judges_dict:
        print(f"ERROR: LLM '{judge_name}' not found. Skipping Process.")
        return
    else:
        LLM_judge = judges_dict[judge_name]


    chatbots_response = csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group1.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group2.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group3.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group4.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group5.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group6.csv') + csv_to_tuple_list(f'{curr_dir}/lmsys_prefs_csv/group7.csv') 

    os.makedirs(f'{curr_dir}/judge_data_agg', exist_ok=True)


    #loads contest data into a list of dictionaries
    if not os.path.isfile(f'{curr_dir}/judge_data_agg/{judge_name}_contest_data.json'):
        contest_data = []
        write_list_to_json(contest_data, f'{curr_dir}/judge_data_agg/{judge_name}_contest_data.json')
    else:
        contest_data = read_json_to_list(f'{curr_dir}/judge_data_agg/{judge_name}_contest_data.json')

    #initializes the judge model
    print(f"Initializing LLM Judge '{judge_name}'.\n")
    llm = LLM_judge(models_folder, device='cuda')
    

    print(f"Initiating Pairwise Comparison for '{judge_name}.'\n")

    for idx, a, b, q, resp_a, resp_b in chatbots_response:

        if {"first": a, "second": b, "question": idx, "winner": a} in contest_data or {"first": a, "second": b, "question": idx, "winner": b} in contest_data or {"first": a, "second": b, "question": idx, "winner": "Tie"} in contest_data:
            pass 

        else:
            try:
                eval_prompt = load_markdown(f'{curr_dir}/reasoning_prompt_judge.md')
                eval_prompt = str(eval_prompt.format(PROMPT=q, ANSWER1=resp_a, ANSWER2=resp_b))
                reasoning_message = [{'role':'user', 'content':eval_prompt}]
                reasoning_prompt  = llm.apply_chat_template(reasoning_message, nb_tokens_max=llm.context_size-llm.upper_answer_size)
                reasoning = await llm.generate(reasoning_prompt)
                print("\n####################################################################")
                print("REASONING:", reasoning, "\n")

                eval_prompt2 = load_markdown(f'{curr_dir}/preference_prompt_judge.md')
                eval_prompt2 = str(eval_prompt2.format(REASONING=str(reasoning)))
                preference_message = [{'role': 'user', 'content':eval_prompt2}]
                preference_prompt  = llm.apply_chat_template(preference_message, nb_tokens_max=llm.context_size-llm.upper_answer_size)
                preference = await llm.generate(preference_prompt)
                print("PREFERENCE:", preference, "\n")
                preference = str(preference)[0:10]

                data = {"first": a, "second": b, "question": idx, "winner": 
                    (
                        a if "1" in preference and "2" not in preference and "0" not in preference
                        else (b if "2" in preference and "1" not in preference and "0" not in preference
                        else ("Tie" if "0" in preference and "1" not in preference and "2" not in preference
                        else None))
                    )
                }  

                if data["winner"]:
                    contest_data.append(data)
                    print(f"{data}\t{len(contest_data)} of {len(chatbots_response)} pairs compared for '{judge_name}'.\n")
                    write_list_to_json(contest_data, f'{curr_dir}/judge_data_agg/{judge_name}_contest_data.json')

                else:
                    print("Preference was 'None'. Skipping.")

            except Exception as e:
                print(f"Error during llm.generate(): {e}")

    print(f"Completed Pairwise Comparison for '{judge_name}'.\n")



def read_json_to_list(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def write_list_to_json(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file, indent=4)


def load_markdown(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def csv_to_tuple_list(file_path):
    tuple_list = []
    
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header row
        for row in csvreader:
            tuple_list.append(tuple(str(element) for element in row[:6]))
    
    shuffle(tuple_list)    
    return tuple_list


if __name__ == "__main__":
    asyncio.run(main())

