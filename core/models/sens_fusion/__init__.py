def import_model(model_id):
    if model_id == 0:
        from .fusion_unet import FusionUnet
        print('using FusionUnet model' )
        return FusionUnet
    elif model_id == 1:
        from .fusion_unetv2 import FusionUnet
        print('using FusionUnet v2 model' )
        return FusionUnet
    else:
        raise NotImplementedError