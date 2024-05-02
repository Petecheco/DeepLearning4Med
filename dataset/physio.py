import torch
import pandas as pd
import os
import torchaudio
from torch.utils.data import Dataset


class PhysioDataset(Dataset):
    def __init__(self, root_path, data_type):
        """
        :param data_type: Choose to create the training set or testing set
        :param root_path: Select the path of the dataset
        """
        assert data_type in ['train', 'test'], "data_type must be either 'train' or 'test'"
        self.root_path = os.path.join(root_path, data_type)
        if data_type == 'train':
            self.folder_names = os.listdir(self.root_path)
            self.wavs = []
            self.labels = []
            for folder_dir in self.folder_names:
                self.folder_dirs = os.path.join(self.root_path, folder_dir)
                reference_file_path = os.path.join(self.folder_dirs, 'REFERENCE.csv')
                reference_csv = pd.read_csv(reference_file_path, header=None)
                for file_name in os.listdir(self.folder_dirs):
                    if file_name.endswith('.wav'):
                        number = file_name.split('.')[0]
                        file_path = os.path.join(self.folder_dirs, file_name)
                        file_label = reference_csv.loc[reference_csv[0]==number][1].item()
                        if file_label==-1:
                            self.labels.append(0)
                        else:
                            self.labels.append(1)
                        self.wavs.append(self.load_wav(file_path))
                    else:
                        continue
        else:
            self.folder_names = os.listdir(self.root_path)
            self.wavs = []
            self.labels = []
            for file_name in self.folder_names:
                self.folder_dirs = os.path.join(self.root_path, file_name)
                reference_file_path = os.path.join(self.root_path, 'REFERENCE.csv')
                reference_csv = pd.read_csv(reference_file_path, header=None)
                if file_name.endswith('.wav'):
                    number = file_name.split('.')[0]
                    file_label = reference_csv.loc[reference_csv[0] == number][1].item()
                    if file_label == -1:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                    self.wavs.append(self.load_wav(self.folder_dirs))
                else:
                    continue
            self.wavs = torch.stack(self.wavs)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.wavs[idx], self.labels[idx]


    def load_wav(self, wav_path):
        output, frquency = torchaudio.load(wav_path)
        return output[:,:10611]
    # How to deal with datapoints with different lengths? this needs to be further improved


if __name__ == '__main__':
    dataset = PhysioDataset("E:\\DeepLearningDataset\\PhysioNet","test")

    print("wait here")
