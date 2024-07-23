import openai
import json
import pickle
import random
import csv
import argparse


class GptAPI:
    def __init__(self,
                 origin_train_file,
                 train_cluster_id_file,
                 origin_raw_file,
                 type,
                 crp_cluster_id_file,
                 num_clusters,
                 kmeans,
                 output_dir) -> None:
        openai.api_base = 
        openai.api_key = 

        self.origin_train_file = origin_train_file # 原始train集
        self.train_cluster_id_file = train_cluster_id_file # train集的cluster_id
        self.origin_raw_file = origin_raw_file # 要生成伪文章的数据集
        self.type = type # 要生成伪文章的数据集的类型：train/dev/test
        self.crp_cluster_id_file = crp_cluster_id_file # raw_data对应的cluster_id
        self.num_clusters = num_clusters # 分类数目
        self.kmeans = kmeans
        self.output_dir = output_dir

        self.train_data = None # train集的数据
        self.train_nq_pairs = [] # train集的nq对
        self.dev_data = None
        self.dev_questions = []
        self.test_questions = [] # 用来存放dev/test两种

        self.train_cluster_id_list = [] # train的分类列表
        self.raw_cluster_id_list = [] # 另一个的分类列表 dev/test
        self.train_clustered_pairs = {} # train的分类映射
        self.selected_pairs = [] # 挑选出来的nq对


    def read_origin_file(self):
        # 读取train的nq对，train/dev/test都会用
        with open(self.origin_train_file) as f:
            self.train_data = json.load(f)
        for i in range(len(self.train_data)):
            self.train_nq_pairs.append({
                "question": self.train_data[i]['question'],
                "passage": self.train_data[i]['positive_ctxs'][0]['text']
            })
        # 读取raw_file
        # dev
        if self.type == 'dev':
            with open(self.origin_raw_file) as f:
                self.dev_data = json.load(f)
            for i in range(len(self.dev_data)):
                self.dev_questions.append(self.dev_data[i]['question'])
        # test
        elif self.type == 'test':
            with open(self.origin_raw_file) as f:
                tsv_reader = csv.reader(f, delimiter='\t')
                for row in tsv_reader:
                    self.test_questions.append(row[0])

    def get_cluster_list(self):
        # 读取train的cluster_list，train/dev/test都会用
        with open(self.train_cluster_id_file, 'rb') as file:
            train_loaded_list = pickle.load(file)
        self.train_cluster_id_list = train_loaded_list.tolist()

        if self.type == 'dev' or self.type == 'test':
            with open(self.crp_cluster_id_file, 'rb') as f:
                raw_loaded_list = pickle.load(f)
            self.raw_cluster_id_list = raw_loaded_list.tolist()


    def get_clustered_pairs(self):
        # 构造train的clustered_pairs，train/dev/test都会用
        for i, cluster in enumerate(self.train_cluster_id_list):
            if cluster not in self.train_clustered_pairs:
                self.train_clustered_pairs[cluster] = []
            self.train_clustered_pairs[cluster].append(self.train_nq_pairs[i])


    def generate_random_number(self, start, end, exclude):
        numbers = [num for num in range(start, end + 1) if num != exclude]
        if numbers:
            return random.choice(numbers)
        else:
            raise ValueError("Exclude number is not within the specified range.")

    def get_selected_pairs(self):
        if self.type == 'train':
            for pair in self.train_nq_pairs:
                cluster = self.train_cluster_id_list[self.train_nq_pairs.index(pair)]

                if self.kmeans == 'topic' or self.kmeans == 'zero_shot':
                    other_pairs_in_cluster = [p for p in self.train_clustered_pairs[cluster] if p != pair]
                    random_pairs = random.sample(other_pairs_in_cluster, min(2, len(other_pairs_in_cluster)))
                    self.selected_pairs.append({"current_pair": pair["question"],
                                                "random_pairs": random_pairs})
                elif self.kmeans == 'kmeans_random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair["question"],
                            "random_pairs": [random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0],
                                             random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0]
                                            ]
                        }
                    )
                elif self.kmeans == 'random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair["question"],
                            "random_pairs": random.sample(self.train_nq_pairs, 2)
                        }
                    )
        elif self.type == 'dev':
            for pair in self.dev_questions:
                cluster = self.raw_cluster_id_list[self.dev_questions.index(pair)]

                if self.kmeans == 'topic' or self.kmeans == 'zero_shot':
                    # raw 和 train不会重复
                    random_pairs = random.sample(self.train_clustered_pairs[cluster], min(2, len(self.train_clustered_pairs[cluster])))
                    self.selected_pairs.append({"current_pair": pair,
                                                "random_pairs": random_pairs})
                elif self.kmeans == 'kmeans_random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair,
                            "random_pairs": [random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0],
                                             random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0]
                                            ]
                        }
                    )
                elif self.kmeans == 'random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair,
                            "random_pairs": random.sample(self.train_nq_pairs, 2)
                        }
                    )
        elif self.type == 'test':
            for pair in self.test_questions:
                cluster = self.raw_cluster_id_list[self.test_questions.index(pair)]

                if self.kmeans == 'topic' or self.kmeans == 'zero_shot':
                    # test 和 train不重复
                    random_pairs = random.sample(self.train_clustered_pairs[cluster], min(2, len(self.train_clustered_pairs[cluster])))
                    self.selected_pairs.append({"current_pair": pair,
                                                "random_pairs": random_pairs})
                elif self.kmeans == 'kmeans_random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair,
                            "random_pairs": [random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0],
                                             random.sample(self.train_clustered_pairs[self.generate_random_number(0, self.num_clusters, cluster)], 1)[0]
                                            ]
                        }
                    )
                elif self.kmeans == 'random':
                    self.selected_pairs.append(
                        {
                            "current_pair": pair,
                            "random_pairs": random.sample(self.train_nq_pairs, 2)
                        }
                    )



    def get_35_api_stream(self, messages: list):
        """为提供的对话消息创建新的回答 (流式传输)

        Args:
            messages (list): 完整的对话消息
            api_key (str): OpenAI API 密钥

        Returns:
            tuple: (results, error_desc)
        """
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages
            )
            return (True, response)
        except Exception as err:
            return (False, f'OpenAI API 异常: {err}')

    def get_nq_d_plust(self):
        with open(self.output_dir, 'a+') as nq_d_plus:
            tsv_writer = csv.writer(nq_d_plus, delimiter='\t', escapechar='$', quoting=csv.QUOTE_NONE)
            tsv_writer.writerow(['id', 'question', 'd+'])

            # 此时selected_pairs里面的current_pair全部是question
            for i in range(self.selected_pairs):
                if self.kmeans == 'topic':
                    messages = [
                        {
                        "role": "user",
                        "content": f"""
                        Please write a passage that answers the given question, here are some examples:\n
                        Example Question1:{self.selected_pairs[i]['random_pairs'][0]['question']},
                        Example Passage1:{self.selected_pairs[i]['random_pairs'][0]['passage']}, \n
                        Example Question2:{self.selected_pairs[i]['random_pairs'][1]['question']},
                        Example Passage2:{self.selected_pairs[i]['random_pairs'][1]['passage']}. \n
                        Question:{self.selected_pairs[i]['current_pair']}
                        """,
                        }
                    ]
                elif self.kmeans == 'zero_shot':
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                            Please write a passage that answers the given question: \n
                            Question:{self.selected_pairs[i]['current_pair']}
                            """,
                        },
                    ]
                else:
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                            Please write a passage that answers the given question, here are some examples:\n
                            Example Question1:{self.selected_pairs[i]['random_pairs'][0]['question']},
                            Example Passage1:{self.selected_pairs[i]['random_pairs'][0]['passage']}, \n
                            Example Question2:{self.selected_pairs[i]['random_pairs'][1]['question']},
                            Example Passage2:{self.selected_pairs[i]['random_pairs'][1]['passage']}. \n
                            Now, write a passage based on the this question: \n
                            Question:{self.selected_pairs[i]['current_pair']}
                            """,
                        },
                    ]

                err_info, completion = self.get_35_api_stream(messages=messages)
                while not err_info:
                    print("Retrying......")
                    err_info, completion = self.get_35_api_stream(messages=messages)

                print(completion['choices'][0]['message']['content'])
                print(f"--------Api processed {i+1} qusetion, {len(self.selected_pairs)-i-1} question to go!--------")
                tsv_writer.writerow([i + 1, self.selected_pairs[i]['current_pair'], completion['choices'][0]['message']['content']])
