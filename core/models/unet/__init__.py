
def import_model(model_id):
    if model_id == 0:
        print('using baseline_unet model')
        from .baseline_unet import UnetModel
        return UnetModel
    elif model_id == 1:
        from .dilated_unet import UnetModel
        return UnetModel
    elif model_id == 2:
        print('using unet_stack_more model')
        from .unet_stack_more import UnetModel
        return UnetModel
    else:
        raise NotImplementedError