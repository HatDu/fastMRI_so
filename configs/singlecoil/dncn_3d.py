model = dict(
    name = 'CRNN_MRI',
    id = 'CRNN_MRI',
    params = dict(
        n_ch=2, nf=64, ks=3, nc=5, nd=5
    )
)

acquisition = ['CORPD_FBK', 'CORPDFS_FBK']
batch_size = 1
data = dict(
    train = dict(
        dataset = dict(
            name = 'data_3d',
            params = dict(
                root='data/singlecoil_train', 
                challenge='singlecoil', 
                sample_num=-1,
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
            name = 'transform_dncn_3d',
            params = dict(
                resolution=320, 
                which_challenge='singlecoil', 
                use_seed=False, 
                crop=False, 
                crop_size=160
            )
        ),
        loader = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    ),
    val = dict(
        dataset = dict(
            name = 'data_3d',
            params = dict(
                root='data/singlecoil_val', 
                challenge='singlecoil', 
                sample_num=50,
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
            name = 'transform_dncn_3d',
            params = dict(
                resolution=320, 
                which_challenge='singlecoil',
                use_seed=True, 
                crop=False, 
                crop_size=96
            )
        ),
        loader = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    ),
    test = dict(
        dataset = dict(
            name = 'data_3d',
            params = dict(
                root='data/singlecoil_test_v2', 
                challenge='singlecoil', 
                sample_num=-1,
                acquisition=acquisition
            )
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(center_fractions=[0.08], accelerations=[4]),
        ),
        transform=dict(
            name = 'transform_dncn_3d',
            params = dict(
                resolution=320, 
                which_challenge='singlecoil',
                use_seed=True, 
                crop=False, 
                crop_size=96
            )
        ),
        loader = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    )
)

logdir = 'log/baseline_unet/'

train = dict(
    optimizer = dict(
        name = 'RMSprop', params=dict(lr=1e-3, weight_decay=0.)
    ),
    lr_scheduler=dict(
        name = 'StepLR',
        params=dict(step_size=40, gamma=0.1)
    ),
    loss = dict(name='l1_loss', params=None),
    train_func = dict(name='train_dncn3d', params=None),
    num_epochs=50,
)

infer = dict(
    infer_func = dict(name='infer_dncn3d', params=None)
)