_base_ = 'stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-3d.py'

load_from = '/home/cvlab123/mmaction2/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-b00440d2.pth'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSampleFrames', clip_len=40),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
