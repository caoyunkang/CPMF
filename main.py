import argparse
from patchcore_runner_cpmf import MultiViewPatchCore

def run_3d_ads(args):
    cls = args.category
    backbone_name = args.backbone

    print('=========================================')
    kwargs = vars(args)
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print('=========================================')

    print(f"\n {args.exp_name} \n")
    print(f"\nRunning on class {cls}\n")
    patchcore = MultiViewPatchCore(backbone_name=backbone_name, dataset_path=args.data_path, n_views=args.n_views, no_fpfh=args.no_fpfh,
                                   class_name=cls, root_dir=args.root_dir, exp_name=args.exp_name, plot_use_rgb=args.use_rgb)

    ############## fit ###############
    patchcore.fit()

    ############# evaluate ###########
    patchcore.evaluate(draw=args.draw)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data-path', type=str, default='../datasets/multi_view_uniform_mvtec_3d_anomaly_detection')
    parser.add_argument('--n-views', type=int, default=1)
    parser.add_argument('--no-fpfh', type=str2bool, default=False)
    parser.add_argument('--use-rgb', type=str2bool, default=False)
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--category', type=str, default='bagel')
    parser.add_argument('--root-dir', type=str, default='./results')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--draw', type=str2bool, default=False)

    args = parser.parse_args()

    run_3d_ads(args)
