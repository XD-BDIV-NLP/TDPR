import argparse
from bert_cls import BertCLS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_path', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    bert_cls_utils = BertCLS(args.question_path, args.save_path)

    bert_cls_utils.read_question()
    bert_cls_utils.get_cls_vector()
    bert_cls_utils.save_cls_vector()