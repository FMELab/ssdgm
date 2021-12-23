import torch

from typing import Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils import data

from torch.utils.data import ConcatDataset, Dataset, DataLoader, random_split

from torchvision import transforms

from src.datamodules.datasets.skillcraft import Skillcraft


class SkillcraftDataModule(LightningDataModule):
    """
    LightningDataModule for Skillcraft dataset.
    """

    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 64,
            split_mode: str = "relative",
            num_workers: int = 0,
            pin_memory: bool = False,
    ):
        super().__init__()

        # This line allows us to access intitialization parameters with the 
        # 'self.hparams' attribute. It also ensures that the initialization
        # parameters will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_mode = split_mode
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        #self.transforms = transforms.ToTensor()
        #self.target_transforms = transforms.ToTensor()

        self.data_train_labeled: Optional[Dataset] = None
        self.data_train_unlabeled: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        Skillcraft(self.hparams.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:

        if not self.data_train_labeled and not self.data_train_unlabeled and not self.data_val and not self.data_test:
            data_set = Skillcraft(self.hparams.data_dir)
            self.dims = data_set.data.size()
            n_samples= self.dims[0]

            # TODO: make the proportions parameters or instance variables
            if self.split_mode == "relative":    
                n_samples_train = int(0.8 * n_samples)
                n_samples_test = n_samples - n_samples_train

                n_samples_train_labeled = int(0.05 * n_samples_train)
                n_samples_unlabeled = n_samples_train - n_samples_train_labeled

                n_val_samples = int(0.1 * n_samples_train_labeled)
                n_samples_train_labeled = n_samples_train_labeled - n_val_samples

                split = [n_samples_train_labeled, n_samples_unlabeled, n_val_samples, n_samples_test]

            # TODO: Implement the logic for the case when we want to specify a concrete number of labeleled train samples
            if self.split_mode == "absolute":
                pass            
            
            self.data_train_labeled, self.data_train_unlabeled, self.data_val, self.data_test = random_split(
                dataset=data_set,
                lengths=split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader_labeled = DataLoader(
            dataset=self.data_train_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        loader_unlabeled = DataLoader(
            dataset=self.data_train_unlabeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        return {"labeled": loader_labeled, "unlabeled": loader_unlabeled}

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    dm = SkillcraftDataModule()
    dm.prepare_data()

