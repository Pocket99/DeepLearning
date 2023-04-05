from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
class PneumothoraxDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['ImageId']
        img_path = 'input/' + img_path + '.png'
        #print(img_path)
        img = Image.open(img_path).convert("RGB")

        rle = self.df.iloc[idx]['EncodedPixels']
        label = 0 if rle == '-1' else 1

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}
    

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])