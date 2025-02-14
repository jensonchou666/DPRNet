
from jscv.datasets.matte import ImagesDataset, ZipDataset, DictDataset, SampleDataset, augmentation as A
from jscv.datasets.matte.data_path import DATA_PATH

from torch.utils.data import DataLoader
from torchvision import transforms as T


def praper_dataset(
    dataset_name,
    batch_size=2,
    train_resize=1024,
    val_resize=1024,
    val_sample_nums=50,
    batch_keys=["img", "true_pha"],
    ):
    # Training DataLoader
    dataset_train = DictDataset(ZipDataset([
            ImagesDataset(DATA_PATH[dataset_name]['train']['fgr'], mode='RGB'),
            ImagesDataset(DATA_PATH[dataset_name]['train']['pha'], mode='L'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((train_resize, train_resize), degrees=(-5, 5), 
                                        translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            A.PairRandomBoxBlur(0.1, 5),
            A.PairRandomSharpen(0.1),
            A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True), batch_keys)

    dataloader_train = DataLoader(dataset_train,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  pin_memory=True)
    
    # Validation DataLoader
    dataset_valid = DictDataset(ZipDataset([
            ImagesDataset(DATA_PATH[dataset_name]['valid']['fgr'], mode='RGB'),
            ImagesDataset(DATA_PATH[dataset_name]['valid']['pha'], mode='L'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((val_resize, val_resize), degrees=(-5, 5),
                                        translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True), batch_keys)

    dataset_valid = SampleDataset(dataset_valid, val_sample_nums)
    dataloader_valid = DataLoader(dataset_valid,
                                  pin_memory=True,
                                  batch_size=batch_size)
    return dataset_train, dataloader_train, dataset_valid, dataloader_valid

if __name__ == "__main__":
    dataset_train, dataloader_train, dataset_valid, dataloader_valid = praper_dataset('PPM100')

    for batch in dataloader_train:
        print(batch.keys())