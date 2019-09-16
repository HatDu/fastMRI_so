def import_model(model_id):
    if model_id == 'CRNN_MRI':
        from .model_pytorch import CRNN_MRI
        print('using CRNN_MRI model' )
        return CRNN_MRI
    else:
        raise NotImplementedError