import os
from data.mvtec3d_cpmf import mvtec3d_classes
from multiprocessing import Pool


if __name__ == "__main__":
    classes = mvtec3d_classes()

    color_options = 'UNIFORM'
    data_path = '../datasets/mvtec3d_multi_view'

    n_processes = len(data_path)
    pool = Pool(processes=1)

    # n_views = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    n_views = [27]

    backbone_names = ['resnet_18']

    no_fpfh_list = [False, True]

    for backbone in backbone_names:
        for no_fpfh in no_fpfh_list:
            for n in n_views:
                for cls in classes:
                    exp_name = f'{color_options}_{backbone}_{n}'
                    if not no_fpfh:
                        exp_name = f'{exp_name}_fpfh'
                    sh = f'python main.py --category {cls} --n-views {n} --no-fpfh {no_fpfh} --data-path {data_path} ' \
                         f'--exp-name {exp_name} --use-rgb {False} ' \
                         f'--backbone {backbone}'

                    print(f'exec {sh}')
                    pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()


