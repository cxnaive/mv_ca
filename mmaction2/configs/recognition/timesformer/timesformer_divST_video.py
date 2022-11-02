_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    # cls_head=dict(type='TimeSformerHead', num_classes=11, in_channels=768),
    cls_head=dict(type='TimeSformerHead', num_classes=10, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
# data_root = '/root/mv_ca/dataset_full/clips_v1.5'
# data_root_val = '/root/mv_ca/dataset_full/clips_v1.5'
# ann_file_train = '/root/mv_ca/dataset_full/train_mm.txt'
# ann_file_val = '/root/mv_ca/dataset_full/validate_mm.txt'
# ann_file_test = '/root/mv_ca/dataset_full/validate_mm.txt'
# ann_file_train = '/root/mv_ca/dataset_full/train_mm_random.txt'
# ann_file_val = '/root/mv_ca/dataset_full/validate_mm_random.txt'
# ann_file_test = '/root/mv_ca/dataset_full/validate_mm_random.txt'

data_root = '/root/mv_ca/datasets/train'
data_root_val = '/root/mv_ca/datasets/validate'
ann_file_train = '/root/mv_ca/datasets/train_mm.txt'
ann_file_val = '/root/mv_ca/datasets/validate_mm.txt'
ann_file_test = '/root/mv_ca/datasets/validate_mm.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)


train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Imgaug', transforms=[
        dict(type='GammaContrast', gamma=(0.3, 0.3)),
        dict(type='EnhanceColor'),
        dict(type='EnhanceContrast'),
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Imgaug', transforms=[dict(type='GammaContrast', gamma=(0.3, 0.3))]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Imgaug', transforms=[dict(type='GammaContrast', gamma=(0.3, 0.3))]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005/8/2,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=1e-4,
    nesterov=True)  # this lr is used for 1 gpu
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/timesformer_dark_dataset_rgb_gamma'
gpu_ids = [0]