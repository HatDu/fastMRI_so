import importlib
from torch.utils.data import DataLoader

dataset_dict = {
    'slice_data': ['core.dataset.slice_data']
}

transform_dict = {
    'slice_transform': ['core.dataset.slice_transform'],
}

mask_dict = {
    'mask_cartesian': ['core.dataset.mask_cartesian'],
}

def get_dataset(name):
    if name not in dataset_dict.keys():
        raise 'No such type dataset'
    module = importlib.import_module(dataset_dict[name])
    return module.MRI_Data

def get_transform(name):
    if name not in transform_dict.keys():
        raise 'No such transform type'
    module = importlib.import_module(transform_dict[name])
    return module.DataTransform

def get_mask(name):
    if name not in mask_dict.keys():
        raise 'No such transform type'
    module = importlib.import_module(mask_dict[name])
    return module.MaskFunc

def create_dataset(cfg):
    mask_cfg = cfg.mask
    transform_cfg = cfg.transform
    dataset_cfg = cfg.dataset

    maskfunc = get_mask(mask_cfg.name)(**mask_cfg.params)
    transform = get_transform(transform_cfg)(maskfunc, **transform_cfg.params)
    data = get_dataset(dataset_cfg.name)(transform, **dataset_cfg.params)
    return data

def create_data_loader(cfg):
    train_data = create_dataset(cfg.data.train)
    dev_data = create_dataset(cfg.data.val)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        **cfg.data.train.loader
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        **cfg.data.val.loader
    )
    display_loader = DataLoader(
        dataset=display_data,
        **cfg.data.val.loader
    )
    return train_loader, dev_loader, display_loader