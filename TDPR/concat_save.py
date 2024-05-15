import pickle
import pandas as pd
import csv
import json


class ConcatAndSave:
    def __init__(self, hidden_path, d_file_path, origin_file_path, write_path, concat, is_test) -> None:
        self.hidden_path = hidden_path
        self.d_file_path = d_file_path
        self.origin_file_path = origin_file_path
        self.write_path = write_path
        self.concat = concat
        self.is_test = False if is_test == 'False' or is_test == 'false' else True


    def concat_and_save(self):
        # 加载筛选后的hidden_tokens list
        with open(self.hidden_path, 'rb') as f:
            hidden_tokens = pickle.load(f)

        # load d‘ 伪文章
        d_file = pd.read_csv(self.d_file_path, sep='\t', quoting=csv.QUOTE_NONE, index_col=None)

        if not self.is_test:
            # 加载原训练文件
            with open(self.origin_file_path) as f:
                origin_file = json.load(f)

            # 问题扩写
            for i in range(len(origin_file)):
                if self.concat == 'both':
                    origin_file[i]['question'] = origin_file[i]['question'] + " [SEP] " + str(d_file.iloc[i]['d+']) + " [SEP] " + " ".join(hidden_tokens[i][0:20])
                elif self.concat == 'keyword':
                    origin_file[i]['question'] = origin_file[i]['question'] + " [SEP] " + " ".join(hidden_tokens[i][0:20])
                elif self.concat == 'd_plus':
                    origin_file[i]['question'] = origin_file[i]['question'] + " [SEP] " + str(d_file.iloc[i]['d+'])

            # 保存扩写结果
            with open(self.write_path, 'w') as f:
                json.dump(origin_file, f, indent=4, ensure_ascii=False)

        else:
            with open(self.origin_file_path) as f:
                origin_file = csv.reader(f, delimiter='\t')

                with open(self.write_path, 'w') as f:
                    nq_writer = csv.writer(f, delimiter='\t', escapechar='^', quoting=csv.QUOTE_NONE, quotechar='"')
                    for i, row in enumerate(origin_file):
                        if self.concat == 'both':
                            row[0] = row[0] + " [SEP] " + str(d_file.iloc[i]['d+']) + " [SEP] " + " ".join(hidden_tokens[i][0:20])
                        elif self.concat == 'keyword':
                            row[0] = row[0] + " [SEP] " + " ".join(hidden_tokens[i][0:20])
                        elif self.concat == 'd_plus':
                            row[0] = row[0] + " [SEP] " + str(d_file.iloc[i]['d+'])

                        nq_writer.writerow([row[0], row[1]])
