import torch

from typing import List, Optional

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch.utils.data import Dataset, DataLoader, random_split

from src.datamodules.datasets.datasets import *

class BaseDataModule(LightningDataModule):
    """
    LightningDataModule for Skillcraft dataset.
    """
    
    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 64,
            n_samples_train_labeled: int = 500,
            val_proportion: float = 0.1,
            n_samples_test: int = 1000,
            use_unlabeled_dataloader: bool = True,
            train_ssdkl: bool = False,
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

        self.train_ssdkl = train_ssdkl
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        #self.transforms = transforms.ToTensor()
        #self.target_transforms = transforms.ToTensor()

        self.data_train_labeled: Optional[Dataset] = None
        self.data_train_unlabeled: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        self._download_dataset()

    def _download_dataset(self) -> None:
        raise NotImplementedError()

    def _load_dataset(self):
        raise NotImplementedError()

    def setup(self, stage: Optional[str] = None) -> None:

        if not self.data_train_labeled and not self.data_train_unlabeled and not self.data_val and not self.data_test:
            self.dataset = self._load_dataset()
            self.dims = self.dataset.data.size()
            n_samples= self.dims[0]

            n_samples_test = self.hparams.n_samples_test
            n_samples_val = int(self.hparams.val_proportion * self.hparams.n_samples_train_labeled)
            n_samples_train_labeled = self.hparams.n_samples_train_labeled - n_samples_val
            n_samples_train_unlabeled = n_samples - (n_samples_train_labeled + n_samples_val + n_samples_test)

            split_lengths = [
                            n_samples_train_labeled, n_samples_train_unlabeled,
                            n_samples_val, n_samples_test
            ]

            
            # We have to do this to ensure the same batch size for unlabeled and labeled examples in SSDKL
            if self.train_ssdkl:# and self.batch_size == 'None':
                self.batch_size = n_samples_train_labeled         
            
            self.data_train_labeled, self.data_train_unlabeled, self.data_val, self.data_test = random_split(
                                                dataset=self.dataset,
                                                lengths=split_lengths,
                                                generator=torch.Generator().manual_seed(42),
            )
            params_dict = self._calc_standardization_params()

            # The following modifies the underlying dataset for all data subsets (labeled, unlabeled, val, test) ...
            # with the means and standard deviations calculated from the training data.
            # This is possible because all subsets reference the whole dataset and ...
            # merely differ in the .indices attribute which determines which dataset samples ...
            # belong to the corresponding data subset.
            self.dataset.data    = (self.dataset.data - params_dict["mean_f"]) / params_dict["std_f"]
            self.dataset.targets = (self.dataset.targets - params_dict["mean_t"]) / params_dict["std_t"]
    
    def _calc_standardization_params(self):

        indices_labeled = self.data_train_labeled.indices
        indices_unlabeled = self.data_train_unlabeled.indices

        features_labeled = self.dataset.data[indices_labeled]
        features_unlabeled = self.dataset.data[indices_unlabeled]

        features = torch.cat((features_labeled, features_unlabeled), axis=0)

        mean_features = torch.mean(features, axis=0)
        std_features = torch.std(features, axis=0)

        targets = self.dataset.targets[indices_labeled]

        mean_targets = torch.mean(targets, axis=0)
        std_targets = torch.std(targets, axis=0)
 
        return {"mean_f": mean_features, "std_f": std_features, "mean_t": mean_targets, "std_t": std_targets}


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader_labeled = DataLoader(
            dataset=self.data_train_labeled,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        dataloaders = {"labeled": loader_labeled}

        if self.hparams.use_unlabeled_dataloader:
            loader_unlabeled = DataLoader(
                dataset=self.data_train_unlabeled,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )

            dataloaders["unlabeled"] = loader_unlabeled

        return dataloaders

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

class SkillcraftDataModule(BaseDataModule):
    """
    LightningDataModule for Skillcraft dataset.
    """
    
    def _download_dataset(self) -> None:
        Skillcraft(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Skillcraft(root=self.hparams.data_dir)

class ParkinsonDataModule(BaseDataModule):
    """
    LightningDataModule for Parkinson dataset.
    """
    
    def _download_dataset(self) -> None:
        Parkinson(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Parkinson(root=self.hparams.data_dir)

class ElevatorsDataModule(BaseDataModule):
    """
    LightningDataModule for Elevators dataset.
    """
    
    def _download_dataset(self) -> None:
        Elevators(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Elevators(root=self.hparams.data_dir)

class ProteinDataModule(BaseDataModule):
    """
    LightningDataModule for Protein dataset.
    """
    
    def _download_dataset(self) -> None:
        Protein(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Protein(root=self.hparams.data_dir)

class BlogDataModule(BaseDataModule):
    """
    LightningDataModule for Blog dataset.
    """
    
    def _download_dataset(self) -> None:
        Blog(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Blog(root=self.hparams.data_dir)

class CTSliceDataModule(BaseDataModule):
    """
    LightningDataModule for CTSlice dataset.
    """
    
    def _download_dataset(self) -> None:
        CTSlice(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return CTSlice(root=self.hparams.data_dir)

class BuzzDataModule(BaseDataModule):
    """
    LightningDataModule for Buzz dataset.
    """
    
    def _download_dataset(self) -> None:
        Buzz(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Buzz(root=self.hparams.data_dir)

class ElectricDataModule(BaseDataModule):
    """
    LightningDataModule for Electric dataset.
    """
    
    def _download_dataset(self) -> None:
        Electric(root=self.hparams.data_dir, download=True)

    def _load_dataset(self):
        return Electric(root=self.hparams.data_dir)