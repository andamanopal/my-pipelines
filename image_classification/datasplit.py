import argparse
import os, shutil
from tqdm import tqdm


def main(config):
    train_size = config.train_size
    valid_size = config.valid_size
    test_size = config.test_size
    assert train_size + valid_size + test_size == 1, f"Total proportion size should be 1.0 (Your input: {train_size} + {valid_size} + {test_size})"

    dataset_path = config.dataset_path
    save_path = config.save_path

    if os.path.exists(save_path):
        assert config.overwrite, "Warning: The specified path exists. Please flag --overwrite=True to continue"
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    classes = []
    for item in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, item)):
            classes.append(item)

    for c in tqdm(classes):

        class_dir = os.path.join(dataset_path, c)
        images = os.listdir(class_dir)

        num_train = int(train_size * len(images))
        num_valid = int(valid_size * len(images))
        # num_test  = int(test_size * len(images))

        images_dict = dict()
        images_dict['train'] = images[:num_train]
        images_dict['valid'] = images[num_train:num_train + num_valid]
        images_dict['test'] = images[num_train + num_valid:]

        os.makedirs(os.path.join(save_path, 'train', c), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'valid', c), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test', c), exist_ok=True)

        for phase in ['train', 'valid', 'test']:
            for image in images_dict[phase]:
                image_src = os.path.join(class_dir, image)
                image_dst = os.path.join(save_path, phase, c, image)
                shutil.copyfile(image_src, image_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/")
    parser.add_argument("--save_path", type=str, default="split_dataset/")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--valid_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    # parser.add_argument("--sequence", type=bool, default=False)
    parser.add_argument("--overwrite", type=bool, default=False)

    config_ = parser.parse_args()
    main(config_)
