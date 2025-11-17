_base_ = 'stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'

load_from = '/home/cvlab123/mmaction2/checkpoints/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSampleFrames', clip_len=40),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=1,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
