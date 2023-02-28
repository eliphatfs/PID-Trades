import torch
import torch_redstone as rst
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10


class ImageClassificationTask(rst.Task):
    crop_size = 32

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    def torchvision_dataset(self, *, root, train, transform, download):
        raise NotImplementedError

    def data(self):
        ttr = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandomCrop(self.crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        tte = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor()
        ])
        train = self.torchvision_dataset(root='logs', train=True, transform=ttr, download=True)
        test = self.torchvision_dataset(root='logs', train=False, transform=tte, download=True)
        return train, test

    def metrics(self):
        return [rst.CategoricalAcc().redstone(), torch.nn.CrossEntropyLoss().redstone()]


class MNISTTask(ImageClassificationTask):
    torchvision_dataset = MNIST
    num_classes = 10


class CIFAR10Task(ImageClassificationTask):
    torchvision_dataset = CIFAR10
    num_classes = 10
