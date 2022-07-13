# -*- coding: utf-8 -*- 
# @Author:  ZhonghuaYi
# @Date  :  7/12/2022
# @Time  :  8:25 PM

def load_fashion_mnist(data_root):
    import struct
    import numpy as np
    train_img_path = data_root + "fashion_mnist/train-images-idx3-ubyte"
    train_label_path = data_root + "fashion_mnist/train-labels-idx1-ubyte"
    test_img_path = data_root + "fashion_mnist/t10k-images-idx3-ubyte"
    test_label_path = data_root + "fashion_mnist/t10k-labels-idx1-ubyte"
    with open(train_img_path, "rb") as f:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', f.read(16))
        train_images = np.fromfile(f, dtype=np.uint8).reshape(images_num, rows * cols)
    with open(train_label_path, "rb") as f:
        labels_magic, labels_num = struct.unpack('>II', f.read(8))
        train_labels = np.fromfile(f, dtype=np.uint8)
    with open(test_img_path, "rb") as f:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', f.read(16))
        test_images = np.fromfile(f, dtype=np.uint8).reshape(images_num, rows * cols)
    with open(test_label_path, "rb") as f:
        labels_magic, labels_num = struct.unpack('>II', f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)
    return train_images, train_labels, test_images, test_labels


def describe(label):
    description = ["T-shirt", "Trouser", "Pullover", "Dress", "coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return description[label]


if __name__ == '__main__':
    pass
