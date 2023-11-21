import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import torch
import argparse
import logging
import torch.nn.functional as F
from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from utils import pyutils
import ruamel.yaml
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
from util import flatten_temporal_batch_dims
from einops import rearrange
from model import build_model
import time

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="pvt", type=str, help="use resnet50 or pvt-v2 as the visual backbone")
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--weights", default='path for best trained model',type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=True, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)
    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    #config
    config_path = "path for avs yaml"
    with open(config_path) as f:
        yy = ruamel.yaml.YAML(typ='safe', pure=True)
        config= yy.load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args), **vars(cfg)}
    config = argparse.Namespace(**config)
    criterion, postprocessor= build_model(config)

    model = AVSModel.Pred_endecoder(**vars(config))
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model,device_ids=[0]).cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    def to_device(sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.cuda()
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [to_device(s) for s in sample]
        elif isinstance(sample, dict):
            sample = {k: to_device(v) for k, v in sample.items()}
        return sample

    
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        collate_fn=test_dataset.collator_test,
                                                        pin_memory=True)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            video_samples, audio_log, audio_pad, targets, category_list, video_name_list, referred_instance_idx = batch_data 
            imgs = video_samples.tensors
            imgs_pad = video_samples.mask
            imgs = imgs.cuda()
            audio = audio_log
            audio = audio.cuda()
            targets = to_device(targets)
            imgs = rearrange(imgs, 't b c h w -> b t c h w')
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) 
            with torch.no_grad():
                audio_feature = audio_backbone(audio) 
            outputs = model(video_samples, audio_pad, imgs, audio_feature) 
            outputs.pop('aux_outputs', None)
            targets_ = []
            targets_.append(targets)
            outputs, targets = flatten_temporal_batch_dims(outputs, targets_) # out [5,50,56,56]
            pred_is_referred = outputs['pred_is_referred'] # [5,50,2]
            prob = F.softmax(pred_is_referred, dim=-1) # [5,50,2]
            scores = prob[..., 0] #[5,50]
            output = F.interpolate(outputs['pred_masks'], size=(224,224), mode="bilinear", align_corners=False) #[5,50,224,224]
            pred_mask = []
            for score, pred in zip(scores,output):
                max_score_idx = score.argmax()
                pred_out = pred[max_score_idx, :, :]
                pred_mask.append(pred_out)
            pred_mask = torch.stack(pred_mask)

            mask = targets[0]['masks'] # [5,224,224]
            mask = torch.tensor(mask, dtype=torch.float32)
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(pred_mask, mask_save_path, category_list, video_name_list)

            miou = mask_iou(pred_mask, mask) # pred [5,224,224] mask [5,480,480]
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(pred_mask, mask, log_dir)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))

        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))












