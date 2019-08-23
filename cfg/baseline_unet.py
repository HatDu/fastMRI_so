data = dict(
    train = dict(
        dataset = dict(
            root='data/', 
            challenge='multicoil', 
            sample_rate=1.
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(
                center_fractions=[0.04, 0.08], 
                accelerations=[8, 4]
            )
        ),
        transform=dict(
            name = 'slice_transform',
            params = dict(
                resolution=320, 
                which_challenge='multicoil', 
                train=True, 
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
    ),
    val = dict(
        dataset = dict(
            root='data/', 
            challenge='multicoil', 
            sample_rate=1.
        ),
        mask=dict(
            name = 'mask_cartesian',
            params = dict(
                center_fractions=[0.04, 0.08], 
                accelerations=[8, 4]
            )
        ),
        transform=dict(
            name = 'slice_transform',
            params = dict(
                resolution=320, 
                which_challenge='multicoil', 
                train=True, 
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