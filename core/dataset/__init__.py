import importlib
from torch.utils.data import DataLoader

dataset_dict = {
    'data_slice': ['core.dataset.data_slice',],
    'data_slicev2': ['core.dataset.data_slicev2',],
    'data_split_multicoil': ['core.dataset.data_split_multicoil',],
    'data_slice_share': ['core.dataset.data_slice_share',],
    'data_slice_adj_2d': ['core.dataset.single_coil.data_slice_adj_2d',],
    'data_3d': ['core.dataset.single_coil.data_3d',],
}

transform_dict = {
    'transform_slice': ['core.dataset.transform_slice',],
    'transform_slice_wo_rss': ['core.dataset.transform_slice_wo_rss',],
    'transform_complex': ['core.dataset.transform_complex',],
    'transform_slice_share': ['core.dataset.transform_slice_share',],
    'transform_dncn': ['core.dataset.single_coil.transform_dncn',],
    'transform_adj_2d': ['core.dataset.single_coil.transform_adj_2d',],
    'transform_dncn_3d': ['core.dataset.single_coil.transform_dncn_3d',],
}

mask_dict = {
    'mask_cartesian': ['core.dataset.mask.mask_cartesian',],
    'mask_cartesian2d': ['core.dataset.mask.mask_cartesian2d',],
}

def get_dataset(name):
    if name not in dataset_dict.keys():
        raise 'No such type dataset'
    module = importlib.import_module(dataset_dict[name][0])
    return module.MRI_Data

def get_transform(name):
    if name not in transform_dict.keys():
        raise 'No such transform type'
    module = importlib.import_module(transform_dict[name][0])
    return module.DataTransform

def get_mask(name):
    if name not in mask_dict.keys():
        raise 'No such transform type'
    module = importlib.import_module(mask_dict[name][0])
    return module.MaskFunc

def create_dataset(cfg, acquisition=None):
    mask_cfg = cfg.mask
    transform_cfg = cfg.transform
    dataset_cfg = cfg.dataset
    if acquisition is not None:
        dataset_cfg.params.acquisition = acquisition
    maskfunc = get_mask(mask_cfg.name)(**mask_cfg.params)
    transform = get_transform(transform_cfg.name)(maskfunc, **transform_cfg.params)
    data = get_dataset(dataset_cfg.name)(transform, **dataset_cfg.params)
    return data

def create_infer_dataloader(cfg):
    data_cfg = cfg.data.test
    mask_cfg = data_cfg.mask
    transform_cfg = data_cfg.transform
    dataset_cfg = data_cfg.dataset
    dataset_cfg.params.acquisition = cfg.acquisition
    mask_func = None
    if cfg.mask:
        mask_func = get_mask(mask_cfg.name)(**mask_cfg.params)
    transform = get_transform(transform_cfg.name)(mask_func, **transform_cfg.params)
    data = get_dataset(dataset_cfg.name)(transform, **dataset_cfg.params)
    data_loader = DataLoader(
        dataset=data,
        **cfg.data.test.loader
    )
    return data_loader

def create_train_dataloaders(cfg):
    train_data = create_dataset(cfg.data.train, cfg.acquisition)
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