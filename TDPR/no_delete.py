from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import csv
import pandas as pd
import string
from collections import OrderedDict
import pickle
import json

class Filter:
    def __init__(self, dpr_file_path, origin_file_path,
                d_plus_path, output_path, is_test) -> None:
        self.model_name = '/data/models/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.mlm_model = BertForMaskedLM.from_pretrained(self.model_name)
        self.stop_words = set(stopwords.words('english'))
        self.punctuation_set = set(string.punctuation)
        self.predicted_words = []

        self.dpr_file_path = dpr_file_path
        self.origin_file_path = origin_file_path
        self.d_plus_path = d_plus_path
        self.output_path = output_path
        self.is_test = is_test

    def remove_special_characters(self, word_list):
        # 去除列表中每个单词的特殊字符
        cleaned_list = [re.sub(r'[^A-Za-z0-9]', '', word) for word in word_list]
        # 过滤掉特殊字符后为空的单词
        cleaned_list = [word for word in cleaned_list if word]
        # 去掉停用词
        cleaned_list = [word for word in cleaned_list if word.lower() not in self.stop_words]

        return cleaned_list

    def filter_and_save_hidden(self):

        dpr_vector = np.load(self.dpr_file_path, allow_pickle=True)
        outputs = []

        for i in range(len(dpr_vector)):
            tensor_input = torch.from_numpy(dpr_vector[i][1]).float().view(1, -1)

            output = self.mlm_model.cls(tensor_input)

            outputs.append(output)

        if self.is_test:
            with open(self.origin_file_path) as test:
                tsv_reader = csv.reader(test, delimiter='\t')
                test_questions = []
                for row in tsv_reader:
                    test_questions.append(row[0])
        else:
            with open(self.origin_file_path, 'rb') as f:
                origin_file = json.load(f)

            # 筛选golden passage
            golden_passages = []

            for i in range(len(origin_file)):
                positive_ctxs = origin_file[i]['positive_ctxs']
                golden_passage = [ctx['text'] for ctx in positive_ctxs]
                golden_passages.append(golden_passage)

        d_file = pd.read_csv(self.d_plus_path, sep='\t', quoting=csv.QUOTE_NONE, index_col=None)
        d_plus = []
        for i in range(len(d_file)):
            if type(d_file.iloc[i]['d+']) is not float:
                d = [text.strip(''.join(self.punctuation_set)) for text in d_file.iloc[i]['d+'].split(' ')]
            else:
                d = []
            d_plus.append(d)

        questions = []
        for i in range(len(d_file)):
            questions.append([text.strip(''.join(self.punctuation_set)) for text in d_file.iloc[i]['question'].split(' ')])

        for i in range(len(outputs)):
            top_k_indices = torch.topk(outputs[i], k=1000).indices

            predicted_word = [self.tokenizer.convert_ids_to_tokens(idx) for idx in top_k_indices.tolist()][0]
            predicted_word = self.remove_special_characters(predicted_word)

            # # 使用OrderDict 保持顺序
            # result_dict = OrderedDict.fromkeys(predicted_word)

            # # 去除第二组列表和第三组字符串中已存在的元素
            # if self.is_test:
            #     for item in [test_questions[i]] + d_plus[i]:
            #         result_dict.pop(item, None)
            # else:
            #     for key in golden_passages[i]:
            #         if key in golden_passages[i]:
            #             del result_dict[key]
            #     # 删除d_plus中存在的键值对
            #     if d_plus[i] != []:
            #         for key in d_plus[i]:
            #             if key in result_dict:
            #                 del result_dict[key]

            #     # 删除问题中存在的键值对
            #     for key in questions[i]:
            #         if key in result_dict:
            #             del result_dict[key]

            if i == 0:
                print(predicted_word)

            self.predicted_words.append(predicted_word)

        with open(self.output_path, 'wb') as f:
            pickle.dump(self.predicted_words, f)

# filter_utils = Filter(dpr_file_path='/data/datasets/WebQuestions/webq_dpr_vector/webq_dpr_test_vector_0',
#                       origin_file_path='/data/datasets/WebQuestions/downloads/data/retriever/qas/webq-test.csv',
#                       d_plus_path='/data/datasets/WebQuestions/d_plus/webq-test-dplus.tsv',
#                       output_path='/home/a8001/cmz_paperwork/wbb_code_test/exp_code/data/raw_webq_test_keywords.pkl',
#                       is_test=False)

filter_utils = Filter(dpr_file_path='/data/datasets/TREC/dpr_embedding/trec_test_0',
                      origin_file_path='/home/a8001/cmz_paperwork/DPR/downloads/data/retriever/qas/curatedtrec-test-origin.csv',
                      d_plus_path='/data/datasets/TREC/d_plus/trec_test_topic.tsv',
                      output_path='/home/a8001/cmz_paperwork/wbb_code_test/exp_code/TREC/data/keywords/trec_test_keywords.pkl',
                      is_test=True)

filter_utils.filter_and_save_hidden()