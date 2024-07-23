import argparse
from kmeans import KMeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cls_path', type=str)
    parser.add_argument('--test_cls_path', type=str)
    parser.add_argument('--dev_cls_path', type=str)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_save_path', type=str)
    parser.add_argument('--dev_save_path', type=str)

    args = parser.parse_args()

    kmeans_utils = KMeans(args.train_cls_path, args.test_cls_path, args.dev_cls_path,
                          args.num_clusters, args.output_path, args.test_save_path,
                          args.dev_save_path)

    kmeans_utils.set_np_seed(123)

    kmeans_utils.axis_process()

    kmeans_utils.set_device()

    kmeans_utils.kmeans_and_save()

    kmeans_utils.get_dev_and_test_list()

    kmeans_utils.save_cluster_list()