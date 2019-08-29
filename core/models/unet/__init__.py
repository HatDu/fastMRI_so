
def import_model(model_id):
    if model_id == 0:
        from .baseline_unet import UnetModel
        return UnetModel
    elif model_id == 1:
        from .dilated_unet import UnetModel
        return UnetModel
    else:
        raise NotImplementedError