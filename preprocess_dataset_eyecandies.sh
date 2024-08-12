origin_data_dir=../../datasets/eyecandies
data_dir=../../datasets/eyecandies_preprocessed
save_dir=../../datasets/eyecandies_multiview
export DISPLAY=:0
cd utils

# remove the background
python preprocessing_eyecandies.py --dataset_path $origin_data_dir --target_path $data_dir

# note: if you run on a cluster without screen, you may need to complie headless open3d:
# http://www.open3d.org/docs/release/tutorial/visualization/headless_rendering.html

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

for class_name in "${class_names[@]}"
    do
        python generate_multi_view_dataset.py --dataset_path $data_dir --color-option UNIFORM --category $class_name --save-dir $save_dir
    done
cd ..