import json
import pickle
import random
import csv
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('/dev/shm/LLama-13b-chat-hf',device_map=device_map,torch_dtype=torch.float16,trust_remote_code=True,use_flash_attention_2=False)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('/dev/shm/LLama-13b-chat-hf',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

parser = argparse.ArgumentParser()
parser.add_argument('--origin_file', type=str)
parser.add_argument('--cluster_id_file', type=str)
parser.add_argument('--kmeans', type=str, help="Using zero_shot, random, kmeans_random or topic.")
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()

with open(args.origin_file) as f:
    nq_data = json.load(f)

# 构建问题-文章对

nq_pairs = []

for i in range(len(nq_data)):
    nq_pairs.append({"question":nq_data[i]['question'], "passage":nq_data[i]['positive_ctxs'][0]['text']})

# 加载问题聚类的列表
# 从文件加载列表
with open(args.cluster_id_file, 'rb') as file:
    loaded_list = pickle.load(file)

# nq_cluster_list = loaded_list.tolist()
nq_cluster_list = loaded_list

# 根据聚类构建一个映射，将每个聚类中的问题-文章对分组
clustered_pairs = {}
for i, cluster in enumerate(nq_cluster_list):
    if cluster not in clustered_pairs:
        clustered_pairs[cluster] = []
    clustered_pairs[cluster].append(nq_pairs[i])


# 遍历每个问题-文章对，选择它所属聚类中的两个随机对
selected_pairs = []

for pair in nq_pairs:
    cluster = nq_cluster_list[nq_pairs.index(pair)]

    if args.kmeans=="topic" or args.kmeans=='zero_shot':
        other_pairs_in_cluster = [p for p in clustered_pairs[cluster] if p != pair]
        random_pairs = random.sample(other_pairs_in_cluster, min(2, len(other_pairs_in_cluster)))
        selected_pairs.append({"current_pair": pair, "random_pairs": random_pairs})
    
    elif args.kmeans=='kmeans_random':
        def generate_random_number(start, end, exclude):
            numbers = [num for num in range(start, end + 1) if num != exclude]
            if numbers:
                return random.choice(numbers)
            else:
                raise ValueError("Exclude number is not within the specified range.")
        selected_pairs.append({"current_pair": pair, 
                                "random_pairs": [random.sample(clustered_pairs[generate_random_number(0,14,cluster)],1)[0],
                                                 random.sample(clustered_pairs[generate_random_number(0,14,cluster)],1)[0]]})

    elif args.kmeans=="random":
        selected_pairs.append({"current_pair": pair, "random_pairs": random.sample(nq_pairs, 2)})
        
# 获取d+文章，并写回tsv文件，[id, question, d+]
# nq_d_plus_name = '/data/datasets/NQ/nq-d-plus-shorter.tsv'
nq_d_plus_name = args.output_dir
    
with open(nq_d_plus_name, 'a+') as nq_d_plus:
    tsv_writer = csv.writer(nq_d_plus, delimiter='\t', escapechar='$', quoting=csv.QUOTE_NONE)
    tsv_writer.writerow(["id", "question", "d+"])
    for i in range(len(selected_pairs)):
        if len(selected_pairs[i]['random_pairs'])>1:
            messages = (f"Please write a passage that answers the given question, here are some examples:\n"
                    f"Example Question1:{selected_pairs[i]['random_pairs'][0]['question']}, \n"
                    f"Example Passage1:{selected_pairs[i]['random_pairs'][0]['passage']}, \n"
                    f"Example Question2:{selected_pairs[i]['random_pairs'][1]['question']},"
                    f"Example Passage2:{selected_pairs[i]['random_pairs'][1]['passage']}. \n"
                    f"Question:{selected_pairs[i]['current_pair']['question']}, \n"
                    f"Passage:\n")
        else:
            print(len(selected_pairs[i]['random_pairs']))
            messages=(f"Please write a passage that answers the given question,\n"
                    f"{selected_pairs[i]['current_pair']['question']}"
                    )
        input_ids = tokenizer([messages], return_tensors="pt",add_special_tokens=False).input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":128,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.4,
            "repetition_penalty":1.3,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id
        }
        generate_ids  = model.generate(**generate_input)
        text = tokenizer.decode(generate_ids[0])
        print(text)
        print(f"--------Api processed {i+1} qusetion, {len(selected_pairs)-i-1} question to go!--------")
        tsv_writer.writerow([i+1, selected_pairs[i]['current_pair']['question'], text.split('\n')[-1]])