import argparse
import glob
import os

import pyarrow.parquet as pq


def convert(parquet_dir, out_dir, synset_file, num_imgs_per_class=10):
    """
    Convert the ImageNet validation set downloaded from Hugging Face (ILSVRC/imagenet-1k, parquet files)
    into one folder per class, named by its synset (n01440764, ...), as expected by demo.py and evaluate.py.

    The first num_imgs_per_class images of each class go to out_dir/X (to compute the universal perturbation,
    as in the paper: 10 images per class = 10 000 images), the others to out_dir/eval (to evaluate it).
    """
    # The Hugging Face labels 0..999 follow the standard ImageNet order (sorted synsets, as in synset_words.txt)
    synsets = [line.split(' ', 1)[0] for line in open(synset_file).read().splitlines() if line]
    files = sorted(glob.glob(os.path.join(parquet_dir, '**', 'validation-*.parquet'), recursive=True))
    if not files:
        raise FileNotFoundError('No validation-*.parquet file in ' + parquet_dir)

    count = [0] * len(synsets)
    for f in files:
        print('Converting', f)
        table = pq.read_table(f, columns=['image', 'label'])
        for image, label in zip(table.column('image').to_pylist(), table.column('label').to_pylist()):
            split = 'X' if count[label] < num_imgs_per_class else 'eval'
            folder = os.path.join(out_dir, split, synsets[label])
            os.makedirs(folder, exist_ok=True)
            name = os.path.basename(image.get('path') or '') or 'img_%05d.JPEG' % count[label]
            with open(os.path.join(folder, name), 'wb') as out:
                out.write(image['bytes'])
            count[label] += 1

    print('>> %d images, %d classes' % (sum(count), sum(c > 0 for c in count)))
    print('>> %s/X : %d images, %s/eval : %d images'
          % (out_dir, sum(min(c, num_imgs_per_class) for c in count),
             out_dir, sum(max(c - num_imgs_per_class, 0) for c in count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the Hugging Face ImageNet validation set to class folders')
    parser.add_argument('parquet_dir', help='folder where the parquet files were downloaded')
    parser.add_argument('out_dir', help='output folder (will contain X/ and eval/)')
    parser.add_argument('--synsets', default='data/synset_words.txt')
    parser.add_argument('--num_imgs_per_class', type=int, default=10, help='images per class kept in X/')
    args = parser.parse_args()
    convert(args.parquet_dir, args.out_dir, args.synsets, args.num_imgs_per_class)
