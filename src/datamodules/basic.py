import os
from itertools import cycle, islice
from typing import List, Optional

import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import VisionDataset
from pathlib import Path
import math

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(file: Path):
    return file.suffix in IMG_EXTENSIONS


def make_dataset(dir, max_dataset_size=float("inf")) -> List[Path]:
    images = []
    root = Path(dir)
    assert root.is_dir(), "%s is not a valid directory" % dir

    for file in root.rglob("*"):
        if is_image_file(file):
            images.append(file)
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        return_paths=False,
        return_dict=False,
        sort=False,
        loader=default_loader,
    ):
        imgs = make_dataset(root)
        if sort:
            imgs = sorted(imgs)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.return_dict = return_dict
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, str(path)
        else:
            if self.return_dict:
                return {"images": img}
            else:
                return img

    def __len__(self):
        return len(self.imgs)


class MergeDataset(data.Dataset):
    def __init__(self, *datasets):
        """Merge multiple datasets to one dataset, and each time retrives a combinations of items in all sub datasets."""
        self.datasets = datasets
        self.sizes = [len(dataset) for dataset in datasets]
        print("dataset size", self.sizes)

    def __getitem__(self, indexs: List[int]):
        return tuple(dataset[idx] for idx, dataset in zip(indexs, self.datasets))

    def __len__(self):
        return max(self.sizes)


class MultiRandomSampler(data.RandomSampler):
    """a Random Sampler for MergeDataset. NOTE will padding all dataset to same length
    Each time it generates an index for each subdataset in MergeDataset.

    Args:
        data_source (MergeDataset): MergeDataset object
        replacement (bool, optional): shuffle index use replacement. Defaults to True.
        num_samples ([type], optional): Defaults to None.
        generator ([type], optional): Defaults to None.
    """

    def __init__(
        self,
        data_source: MergeDataset,
        replacement=True,
        num_samples=None,
        generator=None,
    ):
        self.data_source: MergeDataset = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.maxn = len(self.data_source)

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            self._num_samples = self.data_source.sizes
        return self._num_samples

    def __iter__(self):
        rands = []
        for size in self.num_samples:
            if self.maxn == size:
                rands.append(torch.randperm(size, generator=self.generator).tolist())
            else:
                rands.append(
                    torch.randint(
                        high=size,
                        size=(self.maxn,),
                        dtype=torch.int64,
                        generator=self.generator,
                    ).tolist()
                )
        return zip(*rands)

    def __len__(self):
        return len(self.data_source)


class MultiSequentialSampler(data.Sampler):
    r"""Samples elements sequentially, always in the same order.
        NOTE: it whill expand all dataset to same length

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: MergeDataset):
        self.data_source: MergeDataset = data_source
        self.num_samples = data_source.sizes
        self.maxn = len(data_source)

    def __iter__(self):
        ls = []
        for size in self.num_samples:
            if self.maxn == size:
                ls.append(range(size))
            else:
                ls.append(islice(cycle(range(size)), self.maxn))
        return zip(*ls)

    def __len__(self):
        return len(self.data_source)


class DistributedSamplerWrapper(data.DistributedSampler):
    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            sampler.data_source, num_replicas, rank, shuffle
        )
        self.sampler = sampler

    def __iter__(self):
        indices = list(self.sampler)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return len(self.sampler) // self.num_replicas


class MultiBatchDataset(MergeDataset):
    """MultiBatchDataset for MultiBatchSampler
    NOTE: inputs type must be MergeDataset
    """

    def __getitem__(self, indexs: List[int]):
        dataset_idxs, idxs = indexs
        return self.datasets[dataset_idxs][idxs]


class MultiBatchSampler(data.Sampler):
    r"""Sample another sampler by repeats times of mini-batch indices.
    NOTE always drop last !
      Args:
      samplers (Sampler or Iterable): Base sampler. Can be any iterable object
          with ``__len__`` implemented.
      repeats (list): repeats time
      batch_size (int): Size of mini-batch.

      dataloader是依靠什么停止sample的呢, 是next抛出的error, 还是len
      直接for迭代dataloader时, 是不看len的, 要等到StopIteration
      但pytorch_lighning好像会看len...因此len要和iter一致...

      那么这个len怎么得到呢?
      NOTE: 不同repeat之间必须能够整除...
    """

    def __init__(self, samplers: list, repeats: list, batch_size, drop_last=True):
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )

        assert len(samplers) == len(
            repeats
        ), "Samplers number must equal repeats number"

        minweight = min(
            repeats
        )  # 假设每次sample, 要把频率最小的dataset遍历完, 那么其他的dataset的遍历次数就像最小dataset长度的对应倍数
        minlength = len(samplers[repeats.index(minweight)])
        self.sampler_loop = cycle([i for i, w in enumerate(repeats) for _ in range(w)])
        # expand to target length
        self.repeats = repeats
        self.sizes = [
            minlength * math.ceil(w / minweight) for w in repeats
        ]  # 如果最小的weight是1, 那么其他dataset的size就是minlength的相应倍数
        self.size = sum(self.sizes)
        self.batch_size = batch_size
        self.samplers: List[data.Sampler] = samplers
        self.new_samplers = []
        self.drop_last = True

    def __iter__(self):
        self.new_samplers.clear()
        self.new_samplers = [
            islice(cycle(smp), size)  # size限制了iter的结束点
            for smp, size in zip(self.samplers, self.sizes)
        ]
        return self

    def __next__(self):
        # NOTE sampler_idx choice dataset
        sampler_idx = next(self.sampler_loop)
        sampler: data.Sampler = self.new_samplers[sampler_idx]
        return [
            (sampler_idx, next(sampler)) for _ in range(self.batch_size)
        ]  # 自动droplast, 由于最后一个Batch不满, 会造成next抛出StopIterationn

    def __len__(self):
        # NOTE find min batch scale factor
        scale = (min(self.sizes) // self.batch_size) // min(
            self.repeats
        )  # 算出最先stopiteration的dataset
        return sum([n * scale for n in self.repeats])
