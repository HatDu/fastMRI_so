def import_model(model_id):
    if model_id == 0:
        from .fusion_unet import FusionUnet
        print('using FusionUnet model' )
        return FusionUnet
    else:
        raise NotImplementedError