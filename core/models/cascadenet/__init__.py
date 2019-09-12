def import_model(model_id):
    if model_id == 0:
        from .cascadenet import ComplexNet
        print('using cascadenet model' )
        return ComplexNet
    else:
        raise NotImplementedError