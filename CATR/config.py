from easydict import EasyDict as edict
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "path for vggish pretrained backbone"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "path for resnet50 pretrained backbone"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "path for pvt-v2 pretrained backbone"
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "path for s4_meta_data.csv"
cfg.DATA.DIR_IMG = "path for visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "path for audio_log_mel"
cfg.DATA.DIR_MASK = "path for gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)


if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()
