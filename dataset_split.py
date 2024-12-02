import os
import random
import shutil

image_dir = "./datasets/recognition/images"
label_dir = "./datasets/recognition/labels"

output_dirs = {
    'train': './datasets/recognition/train',
    'val': './datasets/recognition/val',
    'test': './datasets/recognition/test'
}

split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

images = sorted(os.listdir(image_dir))
random.shuffle(images)

split_indices = {
    'train': int(split_ratios['train'] * len(images)),
    'val': int((split_ratios['train'] + split_ratios['val']) * len(images)),
}

splits = {
    'train': images[:split_indices['train']],
    'val': images[split_indices['train']:split_indices['val']],
    'test': images[split_indices['val']:]
}

for split, files in splits.items():
    os.makedirs(os.path.join(output_dirs[split], 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dirs[split], 'labels'), exist_ok=True)

    for file in files:
        # 이미지 이동
        shutil.move(os.path.join(image_dir, file),
                    os.path.join(output_dirs[split], 'images', file))
        # 라벨 이동
        label_file = file.replace('.jpg', '.txt')
        shutil.move(os.path.join(label_dir, label_file),
                    os.path.join(output_dirs[split], 'labels', label_file))