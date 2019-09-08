
def get_infer_func(cfg):
    if cfg.infer.infer_func.name == 'slice':
        from core.infer.infer_slice import run_net, save_reconstructions
        return run_net, save_reconstructions
    if cfg.infer.infer_func.name == 'complex':
        from core.infer.infer_complex import run_net, save_reconstructions
        return run_net, save_reconstructions