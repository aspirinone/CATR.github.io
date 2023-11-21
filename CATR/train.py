import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import random
import torch
import numpy as np
import argparse
import logging
from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import torch.nn.functional as F
from einops import rearrange
import ruamel.yaml
from model import build_model
import torch.cuda.amp as amp
from util import flatten_temporal_batch_dims


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
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=100, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--checkpoint_path', '-ckpt', type=str,
                        help='path of checkpoint file to load for evaluation purposes')

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
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
    config_path = "path for avs.yaml"
    with open(config_path) as f:
        yy = ruamel.yaml.YAML(typ='safe', pure=True)
        config= yy.load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args), **vars(cfg)}
    config = argparse.Namespace(**config)
    model = AVSModel.Pred_endecoder(**vars(config))
    model = torch.nn.DataParallel(model,device_ids=[0]).cuda()
    model.train()

    criterion, postprocessor= build_model(config)
    criterion.cuda()

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        collate_fn=train_dataset.collator,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset('test')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        collate_fn=val_dataset.collator_test,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')

    def to_device(sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.cuda()
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [to_device(s) for s in sample]
        elif isinstance(sample, dict):
            sample = {k: to_device(v) for k, v in sample.items()}
        return sample

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            video_samples, audio_log, audio_pad, targets, referred_instance_idx= batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            imgs = video_samples.tensors
            imgs_pad = video_samples.mask
            imgs = imgs.cuda()
            audio = audio_log
            audio = audio.cuda()
            targets = to_device(targets)
            frame, B,  C, H, W = imgs.shape
            imgs = imgs.reshape(B*frame, C, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]
            with torch.no_grad():
                audio_feature = audio_backbone(audio) # [B*T, 128]

            with amp.autocast(enabled=config.enable_amp):
                outputs = model(video_samples, audio_pad, imgs, audio_feature) # [5,4,50,56,56]
                pred_mask = outputs['pred_masks']
                indices = torch.tensor([0])
                indices = indices.cuda()
                pred_mask = torch.index_select(pred_mask, dim=0, index=indices) # [1,4,50,56,56]
                outputs['pred_is_referred'] = outputs['pred_is_referred'][0].unsqueeze(0)
                outputs['pred_masks'] = pred_mask
                targets_ = []
                targets_.append(targets)
                loss_dict = criterion(outputs, targets_)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


            avg_meter_total_loss.add({'total_loss': losses.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['loss_sigmoid_focal_1']}) 
            avg_meter_sa_loss.add({'sa_loss': loss_dict['loss_is_referred_1']})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            global_step += 1
            if (global_step-1) % 50 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lr: %.4f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), optimizer.param_groups[0]['lr'])

                logger.info(train_log)
                # model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                # torch.save(model.module.state_dict(), model_save_path)
                # print("saving model")


        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
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
                outputs, targets = flatten_temporal_batch_dims(outputs, targets_)
                pred_is_referred = outputs['pred_is_referred'] 
                prob = F.softmax(pred_is_referred, dim=-1) 
                scores = prob[..., 0] 
                output = F.interpolate(outputs['pred_masks'], size=(224,224), mode="bilinear", align_corners=False) #[5,50,224,224]
                pred_mask = []
                for score, pred in zip(scores,output):
                    max_score_idx = score.argmax()
                    pred_out = pred[max_score_idx, :, :]
                    pred_mask.append(pred_out)
                pred_mask = torch.stack(pred_mask)
                mask = targets[0]['masks'] 
                mask = torch.tensor(mask, dtype=torch.float32)
                miou = mask_iou(pred_mask, mask)
                avg_meter_miou.add({'miou': miou})


            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)
            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))


    











