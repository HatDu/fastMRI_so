def import_model(model_id):
    if model_id == 0:
        from .se_unet import SEUnetModel
        return SEUnetModel
    else:
        raise NotImplementedError