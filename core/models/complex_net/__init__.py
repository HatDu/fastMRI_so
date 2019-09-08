def import_model(model_id):
    if model_id == 0:
        from .complex_net import ComplexNet
        print('using ComplexNet model' )
        return ComplexNet
    if model_id == 'unet':
        from .complex_unet import UnetModel
        print('using ComplexNet unet model' )
        return UnetModel
    else:
        raise NotImplementedError