import argparse
from concat_save import ConcatAndSave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_path', type=str)
    parser.add_argument('--d_file_path', type=str)
    parser.add_argument('--origin_file_path', type=str)
    parser.add_argument('--write_path', type=str)
    parser.add_argument('--concat', type=str)
    parser.add_argument('--is_test', type=str)

    args = parser.parse_args()

    concat_save_utils = ConcatAndSave(args.hidden_path, args.d_file_path, args.origin_file_path,
                                      args.write_path, args.concat, args.is_test)
    concat_save_utils.concat_and_save()