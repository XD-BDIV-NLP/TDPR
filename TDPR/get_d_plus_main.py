import argparse
from gpt_api import GptAPI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_train_file', type=str)
    parser.add_argument('--train_cluster_id_file', type=str)
    parser.add_argument('--origin_raw_file', type=int)
    parser.add_argument('--type', type=str)
    parser.add_argument('--crp_cluster_id_file', type=str)
    parser.add_argument('--num_clusters', type=int)
    parser.add_argument('--kmeans', type=str, help='Using zero_shot, random, kmeans_random or topic.')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    api_utils = GptAPI(args.origin_train_file,
                       args.train_cluster_id_file,
                       args.origin_raw_file,
                       args.type,
                       args.crp_cluster_id_file,
                       args.num_clusters,
                       args.kmeans,
                       args.output_dir)


    api_utils.read_origin_file()

    api_utils.get_cluster_list()

    api_utils.get_clustered_pairs()

    api_utils.get_selected_pairs()

    api_utils.get_nq_d_plust()

