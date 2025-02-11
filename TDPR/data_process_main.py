import argparse
from data_process import DataProcess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_train_data_path", type=str)
    parser.add_argument("--origin_dev_data_path", type=str)
    parser.add_argument("--origin_test_data_path", type=str)
    parser.add_argument("train_tsv_save_path", type=str)
    parser.add_argument("dev_tsv_save_path", type=str)
    parser.add_argument("test_tsv_save_path", type=str)
    args = parser.parse_args()

    data_process_utils = DataProcess()

    data_process_utils.read_data(args.origin_train_data_path)
    data_process_utils.csv_writer(args.train_tsv_save_path)

    data_process_utils.read_data(args.origin_dev_data_path)
    data_process_utils.csv_writer(args.dev_tsv_save_path)

    data_process_utils.csv_writer_test(args.origin_test_data_path,
                                       args.test_tsv_save_path)