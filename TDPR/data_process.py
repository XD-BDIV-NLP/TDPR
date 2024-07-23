import json
import csv


class DataProcess:
    def __init__(self) -> None:
        self.origin_data = None

    def read_data(self, path):
        with open(path) as f:
            self.origin_data = json.load(f)

    def csv_writer(self, path):
        with open(path, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t', escapechar='$', quoting=csv.QUOTE_NONE)
            tsv_writer.writerow(["id", "question"])
            for i in range(len(self.data)):
                tsv_writer.writerow([i + 1, self.data[i]["question"]])

    def csv_writer_test(self, path_read, path_write):
        with open(path_read) as nq_t:
            with open(path_write, 'w') as f:
                tsv_writer = csv.writer(f, delimiter='\t')
                tsv_reader = csv.reader(nq_t, delimiter='\t')
                tsv_writer.writerow(["id", "question"])
                for i, row in enumerate(tsv_reader):
                    tsv_writer.writerow([i + 1, row[0]])

