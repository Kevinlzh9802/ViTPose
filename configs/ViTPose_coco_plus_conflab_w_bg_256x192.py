_base_ = [
    './_base_/default_runtime.py',
    # './_base_/datasets/coco.py',
    './_base_/datasets/conflab.py'
]

evaluation = dict(interval=1, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=1, by_epoch=True, max_keep_ckpts=0)

optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=12, 
                                    layer_decay_rate=0.75,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                    )
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# resume
# load_from = 'models/vitpose_base_coco_aic_mpii.pth'
load_from = 'data/conflab/models/vitpose_base_coco_aic_mpii.pth'

# learning policy
lr_config = dict(
    policy='step',
    by_epoch=True,
    step=5)
total_epochs = 5 
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='data/conflab/annotations/'
    'conflab_val_bboxes_xywh.json',
)

coco_data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
    max_num_joints=17,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs',
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs',
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/conflab'
coco_data_root = 'data/coco'
data = dict(
    samples_per_gpu=32, # 64 original value, 12 for 1080 TI, 32 for A40 
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=12), # 32 original value, 6 for 1080 TI, 12 for A40 
    test_dataloader=dict(samples_per_gpu=12), # 32 original value, 6 for 1080 TI, 12 for A40 
    train=[
        # dict(
        #     type='TopDownCocoDataset',
        #     ann_file=f'{coco_data_root}/annotations/person_keypoints_train2017.json',
        #     img_prefix=f'{coco_data_root}/train2017/',
        #     data_cfg=coco_data_cfg,
        #     pipeline=train_pipeline,
        #     # dataset_info={{_base_.dataset_info}}
        #     ),
        dict(
            type='TopDownCocoDataset',
            # ann_file=f'{data_root}/keypoints_and_bboxes_train.json',
            ann_file=f'{data_root}/keypoints_and_bboxes_train_filtered.json',
            img_prefix=f'{data_root}/images_train/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}
            ),    
    ],
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/keypoints_and_bboxes_test.json',
        img_prefix=f'{data_root}/images_test/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}
        ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/keypoints_and_bboxes_test.json',
        img_prefix=f'{data_root}/images_test/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}
        ),
)
