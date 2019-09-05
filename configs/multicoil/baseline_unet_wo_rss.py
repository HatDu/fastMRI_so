model = dict(
    name = 'baseline_unet',
    id = 0,
    params = dict(
        in_chans=15, 
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0
    )
)

acquisition = ['CORPD_FBK', 'CORPDFS_FBK']
data = dict(
    train = dict(
        dataset = dict(
            name = 'data_slice',
            params = dict(
                root='data/multicoil_train', 
                challenge='multicoil', 
                sample_rate=1.,
                acquisition=acquisition
            )
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(
                center_fractions=[0.04, 0.08], 
                accelerations=[8, 4]
            )
        ),
        transform=dict(
            name = 'transform_slice_wo_rss',
            params = dict(
                resolution=320, 
                which_challenge='multicoil', 
                use_seed=True, 
                crop=False, 
                crop_size=160
            )
        ),
        loader = dict(
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    ),
    val = dict(
        dataset = dict(
            name = 'data_slice',
            params = dict(
                root='data/multicoil_val', 
                challenge='multicoil', 
                sample_rate=1.,
                acquisition=acquisition
            )
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(
                center_fractions=[0.04, 0.08], 
                accelerations=[8, 4]
            )
        ),
        transform=dict(
            name = 'transform_slice_wo_rss',
            params = dict(
                resolution=320, 
                which_challenge='multicoil',
                use_seed=True, 
                crop=False, 
                crop_size=96
            )
        ),
        loader = dict(
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    ),
    test = dict(
        dataset = dict(
            name = 'data_slice',
            params = dict(
                root='data/multicoil_test_v2', 
                challenge='multicoil', 
                sample_rate=1.,
                acquisition=acquisition
            )
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(center_fractions=[0.08], accelerations=[4]),
        ),
        transform=dict(
            name = 'transform_slice_wo_rss',
            params = dict(
                resolution=320, 
                which_challenge='multicoil',
                use_seed=True, 
                crop=False, 
                crop_size=96
            )
        ),
        loader = dict(
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    )
)

logdir = 'log/baseline_unet/'

train = dict(
    optimizer = dict(
        name = 'Adam', params=dict(lr=1e-3, weight_decay=0.)
    ),
    lr_scheduler=dict(
        name = 'StepLR',
        params=dict(step_size=40, gamma=0.1)
    ),
    loss = dict(name='l1_loss', params=None),
    train_func = dict(name='train_slice', params=None),
    num_epochs=50,
)

infer = dict(
    infer_func = dict(name='slice', params=None)
)