import os
# Ignore warnings
import warnings

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


class EyePacsDataset(Dataset):
    def __init__(self, dataset_name, root_dir, split, transform):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dataset_dir = os.path.join(self.root_dir, self.dataset_name)
        self.env_dict = {
            'left_nref': 0,
            'right_nref': 1,
            'left_ref': 2,
            'right_ref': 3
        }
        # Checks if the dataset folder exists
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        # Reading the metadata dataframe
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df.loc[self.metadata_df['split'] == split]
        self.y_array = self.metadata_df['level'].values
        self.filename_array = self.metadata_df['image'].values
        self.eye_info_array = self.metadata_df['groups'].values

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        eye_info = self.eye_info_array[idx]
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        return img, y, self.env_dict[eye_info]


def get_eyepacs_dataloader(dataset_name, split, transform, root_dir, batch_size, num_workers):
    kwargs = {'pin_memory': True, 'num_workers': num_workers, 'drop_last': True}
    dataset = EyePacsDataset(dataset_name=dataset_name, root_dir=root_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader


def get_eyepacs_dataset(dataset_name, split, transform, root_dir):
    return EyePacsDataset(dataset_name=dataset_name, root_dir=root_dir, split=split, transform=transform)
