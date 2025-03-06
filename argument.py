import argparse

#设置各种运行参数：
def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', dest='lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling') 

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='hidden dimension') 

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--log', type=str, default='full')

    parser.add_argument('--step-size', type=float, default=4e-3)
    parser.add_argument('--delta', type=float, default=8e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--lncRNA_paths', nargs='+', default=[
        '/root/l-l.csv',
        '/root/l-m.csv',
        '/root/l-p.csv'
    ], help='Paths to lncRNA similarity matrices.')

    parser.add_argument('--go_features_path', type=str, default="GO.csv", help="Path to GO features CSV.")
    parser.add_argument('--l_g_association_path', type=str, default="l-g.csv", help="Path to lncRNA-GO association matrix.")


    return parser.parse_args()

