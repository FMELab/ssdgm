import os
from typing import Any, Callable, Optional, Tuple
from urllib.error import URLError
import torch

from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url
import numpy as np

import pandas as pd

class Skillcraft(Dataset):
  
    mirrors = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00272/"]

    resources = ["SkillCraft1_Dataset.csv"]

    files_to_read = resources

    dataset_file = "skillcraft.csv"


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

        dataset = self.process()
        self._save_data(dataset)

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

        # download the data set file(s)
        # do it in a loop so that other classes can inherit it without problems
        # and use it for a list of multiple resources
        for filename in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    self._download_data(
                        url, root=self.raw_folder
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
    
    def _download_data(self, url, root):
        download_url(url, root)
    
    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]))
        
        dataset = dataset.drop(labels=["GameID"], axis=1)
        
        # get all indices where the entry is `?`
        idx = dataset[
            (dataset["Age"] == '?') |
            (dataset["HoursPerWeek"] == '?') |
            (dataset["TotalHours"] == '?')].index

        dataset = dataset.drop(idx)

        dataset = dataset.astype(np.float32)

        dataset = dataset.drop(dataset[dataset['HoursPerWeek'].isin([0, 168])].index)

        idx = dataset[dataset['HoursPerWeek'] > dataset['TotalHours']].index
        dataset = dataset.drop(idx)

        #features = dataset.loc[:, dataset.columns != "LeagueIndex"].to_numpy()
        #target = dataset.loc[:, "LeagueIndex"].to_numpy()

        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #dataset = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset
        
    def _save_data(self, data) -> None:
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'wb') as f:
        #    np.save(f, data)
        data.to_csv(os.path.join(self.processed_folder, self.dataset_file), index=False)

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy((dataset.loc[:, dataset.columns != "LeagueIndex"]).to_numpy())
        target = torch.from_numpy(dataset.loc[:, "LeagueIndex"].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  

    def _check_raw_exists(self) -> bool:
        raw_folder_exists = os.path.exists(self.raw_folder)
        if not raw_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.raw_folder, filename))
            for filename in self.resources
        )
            
    def _check_processed_exists(self) -> bool:
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return check_integrity(os.path.join(self.processed_folder, self.dataset_file))

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

class Parkinson(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/"]

    resources = ["parkinsons_updrs.data"]

    files_to_read = resources

    target_to_keep = "total_UPDRS"
    target_to_drop = "motor_UPDRS"

    dataset_file = "parkinson.csv"
    
    def _download_data(self, url, root):
        download_url(url, root)

    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]))
        
        dataset.drop(["subject#"], axis=1, inplace=True)
        dataset.drop([self.target_to_drop], axis=1, inplace=True)

        dataset = dataset.astype(np.float32)

        
        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset
    
    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset[dataset.columns[~dataset.columns.isin([self.target_to_keep])]].to_numpy())
        target = torch.from_numpy(dataset[self.target_to_keep].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  

class Elevators(Skillcraft):


    mirrors = ["https://www.dcc.fc.up.pt/~ltorgo/Regression/"]

    resources = ["elevators.tgz"]

    extract_folder = "Elevators"
    files_to_read = ["elevators.data", "elevators.test"]

    dataset_file = "elevators.csv"
    
    
    def _download_data(self, url, root):
        download_and_extract_archive(url, root)

    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        datasets = []
        for file in self.files_to_read:
           ds = pd.read_csv(os.path.join(self.raw_folder, self.extract_folder, file), header=None)
           datasets.append(ds)
        
        dataset = pd.concat(datasets, axis=0)

        dataset = dataset.astype(np.float32)

        
        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #dataset = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, :-1].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, -1].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  

class Protein(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00265/"]

    resources = ["CASP.csv"]

    files_to_read = resources 

    dataset_file = "protein.csv"

    
    def _download_data(self, url, root):
        download_url(url, root)

    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]))

        dataset = dataset.astype(np.float32)

        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, 1:].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, 0].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  

class Blog(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00304/"]

    resources = ["BlogFeedback.zip"]

    files_to_read = ["blogData_train.csv"]

    dataset_file = "blog.csv"


    def _download_data(self, url, root):
        download_and_extract_archive(url=url, download_root=root)
    
    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]), header=None)
        
        dataset = dataset.astype(np.float32)
        # drop all-zero columns (cf. notebook)
        dataset.drop(labels=[12, 32, 37, 277], axis=1, inplace=True)

        
        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, :-1].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, -1].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  

class CTSlice(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00206/"]

    resources = ["slice_localization_data.zip"]

    files_to_read = ["slice_localization_data.csv"]

    dataset_file = "ctslice.csv"

    
    def _download_data(self, url, root):
        download_and_extract_archive(url, root)


    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]))
        
        dataset.drop(['patientId'], axis=1, inplace=True)
        dataset = dataset.astype(np.float32)

        # drop all columns that only have one unique value (i.e. std = 0)
        dataset.drop(labels=["value59", "value69", "value179", "value189", "value351"], inplace=True, axis=1)

        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, :-1].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, -1].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!  


class Buzz(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00248/"]

    resources = ["regression.tar.gz"]

    files_to_read = ["Twitter.data"]

    dataset_file = "buzz.csv"

    
    def _download_data(self, url, root):
        download_and_extract_archive(url, root)

    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(
            os.path.join(
                self.raw_folder,
                self.resources[0].split('.')[0],
                self.files_to_read[0].split('.')[0],
                self.files_to_read[0]),                         
            header=None,
        )

        dataset = dataset.astype(np.float32)        
        
        
        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, :-1].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, -1].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!      


class Electric(Skillcraft):
  
    mirrors = ["https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"]

    resources = ["household_power_consumption.zip"]

    files_to_read = ["household_power_consumption.txt"] 

    dataset_file = "electric.csv"
    
    
    def _download_data(self, url, root):
        download_and_extract_archive(url, root)
        
    def process(self) -> None:
        # process the data set and save it as torch files;
        # to comprehend the processing steps, go to ssdgm/notebooks/skillcraft.ipynb
        dataset = pd.read_csv(os.path.join(self.raw_folder, self.files_to_read[0]), sep=';', low_memory=False)

        dataset.drop(["Date", "Time"], axis=1, inplace=True)
        dataset.dropna(inplace=True)
        dataset = dataset.apply(pd.to_numeric)

        dataset = dataset.astype(np.float32)

    
        
        #dataset = np.concatenate((features, target[:, np.newaxis]), axis=1)
        #data_set = (torch.from_numpy(features), torch.from_numpy(target))

        return dataset

    def _load_data(self):
        #with open(os.path.join(self.processed_folder, self.dataset_file), 'rb') as f:
        #    dataset = np.load(f, allow_pickle=True)
        #data, target = torch.from_numpy(dataset[:, :-1]), torch.from_numpy(dataset[:, -1])
        dataset = pd.read_csv(os.path.join(self.processed_folder, self.dataset_file), dtype=np.float32)

        data = torch.from_numpy(dataset.iloc[:, 1:].to_numpy())
        target = torch.from_numpy(dataset.iloc[:, 0].to_numpy())

        return data, target[:, None]  #! It is essential to do this in each dataset!      

if __name__ == "__main__":
    root = os.path.join(os.getcwd(), "data")
    print(root)
    download = True
    datasets = [
        Skillcraft(root, download=download),
        Parkinson(root, download=download),
        Elevators(root, download=download),
        Protein(root, download=download),
        Blog(root, download=download),
        CTSlice(root, download=download),
        Buzz(root, download=download),
        Electric(root, download=download),
    ]
    for ds in datasets:
        print(f"no. of samples for {ds.dataset_file}: {len(ds)}")