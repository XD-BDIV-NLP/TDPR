import argparse
from filter_save_hidden import Filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpr_file_path', type=str)
    parser.add_argument('--origin_file_path', type=str)
    parser.add_argument('--d_plus_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--is_test', type=str)

    args = parser.parse_args()

    filter_utils = Filter(args.dpr_file_path, args.origin_file_path,
                          args.d_plus_path, args.output_path, args.is_test)

    filter_utils.filter_and_save_hidden()
