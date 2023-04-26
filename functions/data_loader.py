from torchvision import datasets, transforms
import torchvision
import torch
import torch.utils.data as Data


class BaseDataset(Data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.data_length = len(data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.data_length


def load_painting_data():
    data_transfrom = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),             
    ])
    imgs = datasets.ImageFolder('./runs/my_images', transform=data_transfrom)
    # print("read pictures=>")
    # print(len(train_set))
    return imgs
