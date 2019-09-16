
def get_infer_func(cfg):
    if cfg.infer.infer_func.name == 'slice':
        from core.infer.infer_slice import run_net, save_reconstructions
        return run_net, save_reconstructions
    elif cfg.infer.infer_func.name == 'complex':
        from core.infer.infer_complex import run_net, save_reconstructions
        return run_net, save_reconstructions
    elif cfg.infer.infer_func.name == 'infer_dncn':
        from core.infer.infer_dncn import run_net, save_reconstructions
        return run_net, save_reconstructions
    elif cfg.infer.infer_func.name == 'infer_fusion':
        from core.infer.infer_fusion import run_net, save_reconstructions
        return run_net, save_reconstructions
    