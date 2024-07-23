from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import csv

class BertCLS:
    def __init__(self, question_path, save_path) -> None:
        self.model_name = '/data/models/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)

        self.question_path = question_path
        self.save_path = save_path

        self.qdata = []
        self.cls_dict = {}


    def read_question(self):
        with open(self.question_path) as q:
            tsv_reader = csv.reader(q, delimiter='\t')

            # 逐行读取数据
            for row in tsv_reader:
                self.qdata.append(row)

    def get_cls_vector(self):
        for i in range(len(self.qdata)):
            id = i
            # 使用分词器将文本转换为tokens
            tokens = self.tokenizer(self.qdata[i][1], return_tensors="pt")

            # 获取BERT模型的输出
            outputs = self.bert_model(**tokens)

            # 获取CLS向量
            cls_vector = outputs.last_hidden_state[:, 0, :].detach().numpy()
            self.cls_dict[i] = cls_vector

    def save_cls_vector(self):
        np.save(self.save_path, self.cls_dict)
