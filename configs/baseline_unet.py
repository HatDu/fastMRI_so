data = dict(
    train = dict(
        dataset = dict(
            name = 'data_slice',
            params = dict(
                root='data/tmp', 
                challenge='multicoil', 
                sample_rate=1.
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
                crop=True, 
                crop_size=160
            )
        ),
        loader = dict(
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    ),
    val = dict(
        dataset = dict(
            name = 'data_slice',
            params = dict(
                root='data/multicoil_val', 
                challenge='multicoil', 
                sample_rate=1.
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
            num_workers=8,
            pin_memory=True,
        )
    )
)