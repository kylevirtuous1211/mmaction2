_base_ = 'stgcnpp_8xb16-joint-u100-80e_OurDataset-xsub-keypoint-2d.py'

load_from = '/home/cvlab123/mmaction2/checkpoints/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth'
# load_from = '/home/cvlab123/mmaction2/work_dirs/916_rtmpose_all_2D_bone_motion/best_acc_top1_epoch_16.pth'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSampleFrames', clip_len=40),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=40, num_clips=1,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
