import json
import os
from typing import Any, Tuple

import torchvision.datasets
from PIL import Image
from torch.utils.data import Dataset


class GenericDataset(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform, test=False, **kwargs) -> None:
        assert (root is not None) and (transform is not None)
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "test")
        super(GenericDataset, self).__init__(root=split_data_dir, transform=transform)


def read_classnames(classname_file):
    with open(classname_file, "r") as f:
        return [line.strip().split("\t")[1].replace("_", " ") for line in f]


def normalize(name):
    return name.strip().lower().replace("_", " ")


class ImageList(Dataset):
    # (image path, label) samples, labels following the order of classname_file used by the zero-shot classifier

    def __init__(self, samples, classname_file, transform):
        self.samples = samples
        self.classes = read_classnames(classname_file)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


class CoOpSplit(ImageList):
    # Test split of a dataset in the CoOp layout written by vlm-tuning's download_datasets.sh:
    # <root>/<image_dir>/<image path> and <root>/<split_file> = {"test": [[image path, label, class name], ...]}.
    # The split's labels follow CoOp's class order, so samples are relabeled by class name against classname_file.
    image_dir: str
    split_file: str
    classname_file: str
    match_folder = False  # match the class by the image's folder instead of the split's class name

    def __init__(self, transform, root, test=True, **kwargs):
        label_of = {normalize(c): i for i, c in enumerate(read_classnames(self.classname_file))}
        with open(os.path.join(root, self.split_file), "r") as f:
            split = json.load(f)["test"]
        samples = []
        for path, _, name in split:
            key = normalize(path.split("/")[0] if self.match_folder else name)
            if key not in label_of:
                raise KeyError(f"class '{key}' of {self.split_file} is not in {self.classname_file}")
            samples.append((os.path.join(root, self.image_dir, path), label_of[key]))
        super().__init__(samples, self.classname_file, transform)


class Aircraft(ImageList):
    # <root> is the data/ folder of fgvc-aircraft-2013b; classname_file follows the order of variants.txt

    def __init__(self, transform, root, test=True, **kwargs):
        classname_file = "data/classnames/fgvc_aircraft.txt"
        with open(os.path.join(root, "variants.txt"), "r") as f:
            variants = [line.strip() for line in f]
        assert len(variants) == len(read_classnames(classname_file))
        samples = []
        with open(os.path.join(root, "images_variant_test.txt"), "r") as f:
            for line in f:
                image_id, variant = line.strip().split(" ", 1)
                samples.append((os.path.join(root, "images", f"{image_id}.jpg"), variants.index(variant)))
        super().__init__(samples, classname_file, transform)


class Bird(GenericDataset):

    def __init__(self, transform, test=True, **kwargs):
        super(Bird, self).__init__(
            root="/dataset/CUB-200-2011",
            transform=transform,
            test=test,
        )


class Car(CoOpSplit):
    image_dir = ""  # the split's paths start with cars_test/
    split_file = "split_zhou_StanfordCars.json"
    classname_file = "data/classnames/stanford_cars.txt"


class Caltech101(CoOpSplit):
    image_dir = "101_ObjectCategories"
    split_file = "split_zhou_Caltech101.json"
    classname_file = "data/classnames/caltech101.txt"
    match_folder = True  # classname_file holds the folder names (CoOp renames a few, e.g. Faces -> face)


class DTD(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_DescribableTextures.json"
    classname_file = "data/classnames/dtd.txt"


class EuroSAT(CoOpSplit):
    image_dir = "2750"
    split_file = "split_zhou_EuroSAT.json"
    classname_file = "data/classnames/eurosat.txt"


class Food(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_Food101.json"
    classname_file = "data/classnames/food101.txt"


class Flower(CoOpSplit):
    image_dir = "jpg"
    split_file = "split_zhou_OxfordFlowers.json"
    classname_file = "data/classnames/oxford_flowers.txt"


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(self, transform, test=False, root="/dataset/imagenet", **kwargs):
        train = not test
        if train:
            split_data_dir = os.path.join(root, "train")
        else:
            split_data_dir = os.path.join(root, "val")
        super(ImageNet, self).__init__(root=split_data_dir, transform=transform)


class Pet(CoOpSplit):
    image_dir = "images"
    split_file = "split_zhou_OxfordPets.json"
    classname_file = "data/classnames/oxford_pets.txt"


class SUN397(CoOpSplit):
    image_dir = "SUN397"
    split_file = "split_zhou_SUN397.json"
    classname_file = "data/classnames/sun397.txt"


class UCF101(CoOpSplit):
    image_dir = "UCF-101-midframes"
    split_file = "split_zhou_UCF101.json"
    classname_file = "data/classnames/ucf101.txt"
