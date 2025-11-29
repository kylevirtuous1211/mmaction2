ann_file_test = '/home/cvlab123/data/pickle/1101_Ourdataset_with_idle.pkl'
ann_file_train = '/home/cvlab123/data/pickle/1101_Ourdataset_with_idle.pkl'
auto_scale_lr = dict(base_batch_size=128, enable=False)
class_weight = [
    0.97,
    0.78,
    1.06,
    0.96,
    1.08,
    1.02,
    1.1,
    1.04,
    1.03,
    1.03,
]
dataset_type = 'PoseDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='acc/top1',
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
imports = (
    'mmaction.datasets.transforms.pose_transforms',
    'fias_custom_loss.custom_loss',
)
launcher = 'none'
launcher_cfg = dict(env_vars=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8'))
load_from = '/home/cvlab123/mmaction2/checkpoints/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        gcn_adaptive='init',
        gcn_with_res=True,
        graph_cfg=dict(layout='coco', mode='spatial'),
        tcn_type='mstcn',
        type='STGCN'),
    cls_head=dict(
        in_channels=256,
        loss_cls=dict(
            coarse_penalty_weight=2, type='HierarchicalExerciseLoss'),
        num_classes=10,
        type='GCNHead'),
    data_preprocessor=dict(type='mmaction.ActionDataPreprocessor'),
    type='RecognizerGCN')
optim_wrapper = dict(
    optimizer=dict(
        lr=0.001, momentum=0.9, nesterov=True, type='SGD',
        weight_decay=0.0005))
param_scheduler = [
    dict(
        T_max=20,
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=True, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='/home/cvlab123/data/pickle/1101_Ourdataset_with_idle.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=40,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_test',
        test_mode=True,
        type='PoseDataset'),
    num_workers=0,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(type='AccMetric'),
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(clip_len=40, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        dataset=dict(
            ann_file='/home/cvlab123/data/pickle/1101_Ourdataset_with_idle.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(dataset='coco', feats=[
                    'j',
                ], type='GenSkeFeat'),
                dict(clip_len=40, type='UniformSampleFrames'),
                dict(type='PoseDecode'),
                dict(theta=0.3, type='RandomRot'),
                dict(scale=0.1, type='RandomScale'),
                dict(num_person=1, type='FormatGCNInput'),
                dict(type='PackActionInputs'),
            ],
            split='xsub_train',
            type='PoseDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(clip_len=40, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='/home/cvlab123/data/pickle/1101_Ourdataset_with_idle.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(dataset='coco', feats=[
                'j',
            ], type='GenSkeFeat'),
            dict(
                clip_len=40,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(type='PoseDecode'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='xsub_val',
        test_mode=True,
        type='PoseDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(type='AccMetric'),
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(dataset='coco', feats=[
        'jm',
    ], type='GenSkeFeat'),
    dict(clip_len=40, num_clips=1, test_mode=True, type='UniformSampleFrames'),
    dict(type='PoseDecode'),
    dict(num_person=1, type='FormatGCNInput'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/1101_CL_reweighting_add_idle_2D_joint_motion'
