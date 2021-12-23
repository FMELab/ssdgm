import os
from typing import Any, Callable, Optional, Tuple
from urllib.error import URLError
from numpy import float32
import torch
from torch.utils import data

from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url

from sklearn.preprocessing import StandardScaler

import pandas as pd

class Skillcraft(Dataset):
    """

    """
    # Make sure to update the URLs when you change the file in the Google
    # Drive folder!
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv"

    filename = os.path.basename(url)

    data_set_file = "skillcraft.pt"

    scaler = StandardScaler()

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if self._check_processed_exists():
            self.data, self.targets = self._load_data()

        # download the data
        if download:
            self.download()


        if not self._check_processed_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        # load the processed data    
        self.data, self.targets = self._load_data()

    def download(self) -> None:
        """Download the Skillcraft data, if it doesn't exist already."""
        
        if self._check_processed_exists():
            return


        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download the data set file
        download_url(
            url=self.url,
            root=self.raw_folder,
            filename=self.filename
        )

        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        data_set = pd.read_csv(os.path.join(self.raw_folder, self.filename))
        
        data_set = data_set.drop(labels=["GameID"], axis=1)
        
        idx = data_set[
            (data_set["Age"] == '?') |
            (data_set["HoursPerWeek"] == '?') |
            (data_set["TotalHours"] == '?')].index
        data_set = data_set.drop(idx)

        data_set = data_set.astype(float32)

        data_set = data_set.drop(data_set[data_set['HoursPerWeek'].isin([0, 168])].index)

        idx = data_set[data_set['HoursPerWeek'] > data_set['TotalHours']].index
        data_set = data_set.drop(idx)

        data_set = data_set.to_numpy()

        # standardize the whole dataset because the authors of the SSDKL paper have done it like this;
        # normally, we would only standardize the features and do not include the target
        data_set = self.scaler.fit_transform(data_set)

        data_set = torch.from_numpy(data_set)

        data = data_set[:, :-1]
        target = data_set[:, -1]

        data_set = (data, target)

        with open(os.path.join(self.processed_folder, self.data_set_file), "wb") as f:
            torch.save(data_set, f)


    def _load_data(self):
        data, target = torch.load(os.path.join(self.processed_folder, self.data_set_file))

        return data, target 

    def _check_raw_exists(self) -> bool:
        raw_folder_exists = os.path.exists(self.raw_folder)
        if not raw_folder_exists:
            return False

        return check_integrity(os.path.join(self.raw_folder, self.filename))
            
    def _check_processed_exists(self) -> bool:
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return check_integrity(os.path.join(self.processed_folder, self.data_set_file))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index(int): Index

        Returns:
            tuple: (data, target) 
        """
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")


if __name__ == "__main__":
    data_set = Skillcraft("/home/flo/ssdgm/data", download=True)
    print(data_set[43])
    
