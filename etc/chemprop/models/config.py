import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = '../logger'

_C.TAG = 'default'

_C.SEED = 1

_C.NUM_FOLDS = 10

_C.SHOW_EACH_SCORES = False

_C.EVAL_MODE = False

_C.SHOW_FREQ = 5

_C.HYPER = False
_C.HYPER_COUNT = 1
_C.HYPER_REMOVE = ('str', None)

_C.NUM_ITERS = 20

_C.DATA = CN()

_C.DATA.BATCH_SIZE = 8

_C.DATA.DATA_PATH = '../data/'

_C.DATA.DATASET = 'bbbp'

_C.DATA.TASK_NAME = ['p_np']
# 'classification' or 'regression'
_C.DATA.TASK_TYPE = 'classification'
# ['auc', 'prc', 'rmse', 'mae']
_C.DATA.METRIC = 'auc'
# 'random', 'scaffold' or 'noise'
_C.DATA.SPLIT_TYPE = 'random'

_C.MODEL = CN()

_C.MODEL.HID1 = 64
_C.MODEL.HID2 = 64

_C.MODEL.OUT_DIM = 2

_C.MODEL.DEPTH = 3

_C.MODEL.SLICES = 2

_C.MODEL.DROPOUT = 0.2

_C.MODEL.F_ATT = True

_C.MODEL.R = 4

_C.MODEL.BRICS = True
_C.MODEL.MacFrag = True

_C.LOSS = CN()

_C.LOSS.FL_LOSS = False

_C.LOSS.CL_LOSS = True

_C.LOSS.ALPHA = 0.1

_C.LOSS.TEMPERATURE = 0.1

_C.TRAIN = CN()

_C.TRAIN.RESUME = False
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCHS = 10

_C.TRAIN.EARLY_STOP = -1

_C.TRAIN.TENSORBOARD = CN()
_C.TRAIN.TENSORBOARD.ENABLE = True

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'adam'

_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3

_C.TRAIN.OPTIMIZER.FP_LR = 4e-5

_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'reduce'

_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 2.0
_C.TRAIN.LR_SCHEDULER.INIT_LR = 1e-4
_C.TRAIN.LR_SCHEDULER.MAX_LR = 1e-2
_C.TRAIN.LR_SCHEDULER.FINAL_LR = 1e-4

_C.TRAIN.LR_SCHEDULER.FACTOR = 0.7
_C.TRAIN.LR_SCHEDULER.PATIENCE = 10
_C.TRAIN.LR_SCHEDULER.MIN_LR = 1e-5


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(cfg, args):
    _update_config_from_file(cfg, args.cfg)

    cfg.defrost()
    if args.opts:
        cfg.merge_from_list(args.opts)

    if args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if args.lr_scheduler:
        cfg.TRAIN.LR_SCHEDULER.TYPE = args.lr_scheduler
    if args.resume:
        cfg.TRAIN.RESUME = args.resume
    if args.tag:
        cfg.TAG = args.tag
    if args.eval:
        cfg.EVAL_MODE = True

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)

    cfg.freeze()


def get_config(args):

    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
