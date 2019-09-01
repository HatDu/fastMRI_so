def import_model(model_id):
    if model_id == 0:
        from .se_unet import UnetModel
        return UnetModel
    else:
        raise NotImplementedError