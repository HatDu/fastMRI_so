model = dict(
    name = 'baseline_unet',
    id = 0,
    num_stages=10,
    numfilter=24,
    filter_size=11,
    num_weights=31,
    vmax=1.0,
    vmin=-1.0,
    params = dict(
        network=dict(
            num_stages=num_stages,
            pad=11,
            numfilter=numfilter,
            filter_size=filter_size,
            num_weights=num_weights,
            vmin=vmin,
            vmax=vmax,
            datatermweight=1.0
        ),
        reg=dict(
            activation=dict(
                name='w1',
                num_stages=num_stages,
                numfilter=numfilter,
                num_weights=num_weights,
                vmin=vmin,
                vmax=vmax,
                init_type='linear',
                init_scale=0.04
            ),
            filter1=dict(
                name='k1',
                num_stages=num_stages,
                features_in=1,
                filter_out=numfilter,
                filter_size=filter_size,
                prox_zero_mean=True,
                prox_norm=True
            )
        )
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
            name = 'transform_slice',
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
            name = 'transform_slice',
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
            name = 'transform_slice',
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