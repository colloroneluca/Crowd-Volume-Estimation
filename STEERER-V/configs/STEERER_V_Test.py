gpus = (0)
log_dir = 'exp'
workers = 8 #was 12
print_freq = 30
seed = 3035
log = True
val_each_epoch = True
scale = 1000
temporal = False
pretrained = ''
network = dict(
    backbone="MocHRBackbone",
    sub_arch='hrnet48',
    counter_type = 'withMOE', #'withMOE' 'baseline'
    resolution_num = [0,1,2,3],
    loss_weight = [1., 1./2, 1./4, 1./8],
    sigma = [4],
    gau_kernel_size = 15,
    baseline_loss = False,
    pretrained_backbone="../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth",
    
    head = dict(
        type='CountingHead',
        fuse_method = 'cat',
        in_channels=96,
        stages_channel = [384, 192, 96, 48],
        inter_layer=[64,32,16],
        out_channels=1)
)

dataset = dict(
    name='SHHB',
    head=False,
    root='../ProcessedData/anthropos_v/',
    temporal=False,
    test_set='test2.txt',
    train_set='train.txt',
    loc_gt = 'test_gt_loc.txt',
    num_classes= len(network['resolution_num']),
    den_factor=100,
    extra_train_set =None
)


optimizer = dict(
    NAME='adamw',
    BASE_LR=8e-6,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-2,
    EPS= 1.0e-08,
    MOMENTUM= 0.9,
    AMSGRAD = False,
    NESTEROV= True,
)


lr_config = dict(
    NAME='cosine',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=250,
    DECAY_RATE = 0.1,
    WARMUP_EPOCHS=1,   # the number of epochs to warmup the lr_rate
    WARMUP_LR=5.0e-07,
    MIN_LR= 1.0e-07
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

train = dict(
    counter='normal',
    image_size=(512,512),#(768,768),  # height width
    route_size=(256, 256),  # height, width
    base_size=1280,
    batch_size_per_gpu=16,
    shuffle=True,
    begin_epoch=0,
    end_epoch=300,
    extra_epoch=0,
    extra_lr = 0,
    #  RESUME: true
    resume_path= '../PretrainedModels/',
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1.0/0.5),
    val_span = [-1000, -600, -400, -200, -200, -100, -100],
    downsamplerate= 1,
    ignore_label= 255
)


test = dict(
    image_size=(1024, 2048),  # height, width
    base_size=2048,
    loc_base_size=2048,
    loc_threshold=0.20,
    batch_size_per_gpu=1,
    patch_batch_size=2,
    flip_test=False,
    multi_scale=False,
    model_file = ''
)

CUDNN = dict(
    BENCHMARK= True,
    DETERMINISTIC= False,
    ENABLED= True)


