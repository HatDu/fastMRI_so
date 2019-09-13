def import_model(model_id):
    if model_id == 0:
        from .model import DnCn
        print('using DnCn model' )
        return DnCn
    else:
        raise NotImplementedError