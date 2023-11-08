import os
from data.mvtec3d_cpmf import mvtec3d_classes
from multiprocessing import Pool


if __name__ == "__main__":
    classes = mvtec3d_classes()

    dataset_path = '../datasets/mvtec_3d_anomaly_detection'
    color_options = ['UNIFORM']
    save_dirs = ['../datasets/multi_view_uniform_mvtec_3d_anomaly_detection']
    data_path = save_dirs

    n_processes = len(data_path)
    pool = Pool(processes=4)

    n_views = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    backbone_names = ['resnet_18']

    no_fpfh_list = [False, True]

    prefix = color_options

    for backbone in backbone_names:
        for no_fpfh in no_fpfh_list:
            for n in n_views:
                for cls in classes:
                    for p, pre in zip(data_path, prefix):
                        exp_name = f'{pre}_{backbone}_{n}'
                        if pre == 'rgb':
                            use_rgb = True
                        else:
                            use_rgb = False

                        if not no_fpfh:
                            exp_name = f'{exp_name}_fpfh'
                        sh = f'python main.py --category {cls} --n-views {n} --no-fpfh {no_fpfh} --data-path {p} ' \
                             f'--exp-name {exp_name} --use-rgb {use_rgb} ' \
                             f'--backbone {backbone}'

                        print(f'exec {sh}')
                        pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()


