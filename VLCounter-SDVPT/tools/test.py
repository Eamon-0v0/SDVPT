import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import random
import yaml
from dotmap import DotMap
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from .models.VLCounter import Counter
from .dataset.datasets import get_val_loader,get_train_token
from .util import save_density_map, get_model_dir
import sys
import time
from scipy import stats
def parse_args() -> None:
    parser = argparse.ArgumentParser(description='VLCounter')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='gpu ids')
    parser.add_argument('--enc', type=str, required=True, help='SPT encoder setting')
    parser.add_argument('--num_tokens', type=int, help='num of SPT')
    parser.add_argument('--patch_size', type=int, help='patch size')
    parser.add_argument('--prompt', type=str, required=True, help='prompt type')
    parser.add_argument('--ckpt_used', type=str, required=True, help='best checkpoint')
    parser.add_argument('--exp', type=str, required=True, help='exp')

    parsed = parser.parse_args()
    assert parsed.config is not None
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.config = parsed.config
    args.gpus = parsed.gpus
    args.enc = parsed.enc
    args.num_tokens = parsed.num_tokens
    args.patch_size = parsed.patch_size
    args.prompt = parsed.prompt
    args.EVALUATION.ckpt_used = parsed.ckpt_used
    args.exp = parsed.exp

    return args


def main(args):
    if args.TRAIN.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.TRAIN.manual_seed)
        np.random.seed(args.TRAIN.manual_seed)
        torch.manual_seed(args.TRAIN.manual_seed)
        torch.cuda.manual_seed_all(args.TRAIN.manual_seed)
        random.seed(args.TRAIN.manual_seed)

    model = Counter(args).cuda()
    for k, v in model.t_enc.named_parameters():
        if 'prompt' not in k:
            v.requires_grad = False
    for k, v in model.v_enc.named_parameters():
        if 'prompt' not in k and 'text_proj' not in k and 'MHA' not in k:
            v.requires_grad=False

    
    root_model = get_model_dir(args)

    if args.EVALUATION.ckpt_used is not None:
        filepath = os.path.join(root_model, f'{args.EVALUATION.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading model weight '{}'".format(filepath),flush=True)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model weight '{}'".format(filepath),flush=True)
    else:
        print("=> Not loading anything",flush=True)

    test_loader = get_val_loader(args,mode='test')
    # ====== Test  ======
    val_mae,val_rmse =validate_model(
        args= args,
        val_loader=test_loader,
        model=model,
        model_save_dir=os.path.join(root_model,'inference'),
        epoch = 0,
        mode = 'test',
        stage= 'test'   # As long as mode! = 'train' OR 'trian2', it is SDPE    
    )


def validate_model(
    args: argparse.Namespace,
    val_loader: torch.utils.data.DataLoader,
    model : torch.nn.Module,
    model_save_dir : str,
    epoch : int,
    mode : str,
    stage: str
):

    print("===> Start testing")

    model.eval()
    qry_mae = 0
    qry_rmse = 0
    runtime = 0
    qry_nae = 0
    qry_sre = 0

    tokenized_train_text_list=get_train_token(args.prompt)
    t_train=  []
    for tt in tokenized_train_text_list:
        tt=tt.cuda()
        with torch.no_grad():
            _t_train = model.t_enc(tt)
            _t_train = _t_train / _t_train.norm(dim=-1, keepdim=True)
            _t_train = _t_train.mean(dim=0)
            _t_train /= _t_train.norm()
            t_train.append(_t_train)
    _t_train = torch.stack(t_train)

    with torch.no_grad():
        for i, (query_img, query_den, tokenized_text, class_chosen, class_ids) in enumerate(val_loader):
          
            t0= time.time()

            class_ids=class_ids.to('cpu').numpy().tolist() 
            query_img, query_den, tokenized_text = query_img.cuda(), query_den.cuda(), tokenized_text.cuda()
            _, _, h, w = query_img.shape

            density_map = torch.zeros([h, w])
            density_map = density_map.cuda()
            attn_map = torch.zeros([h, w])
            attn_map = attn_map.cuda()
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    output, attn, _,_,_ = model(query_img[:, :, :, start:start + 384], tokenized_text,stage, class_ids,_t_train)
                    output = output.squeeze(0).squeeze(0)
                    attn = attn.squeeze(0).squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    a1 = b1(attn[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])
                    a2 = b2(attn[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    attn_map_l = b3(attn_map[:, 0:start])
                    attn_map_m = b1(attn_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])
                    attn_map_r = b4(attn_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                    attn_map = attn_map_l + attn_map_r + attn_map_m / 2 + a1 / 2 + a2


                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            density_map /= 60.
            pred_cnt = torch.sum(density_map).item()
            gt_cnt = torch.sum(query_den).item()
            cnt_err = abs(pred_cnt-gt_cnt)
            qry_mae += cnt_err
            qry_rmse += cnt_err**2
            qry_nae += cnt_err/gt_cnt
            qry_sre += (cnt_err**2)/gt_cnt


            t1= time.time()
            runtime += (t1-t0)
        qry_mae = qry_mae / len(val_loader.dataset)
        qry_rmse = (qry_rmse/len(val_loader.dataset)) ** 0.5
        qry_nae = qry_nae / len(val_loader.dataset)
        qry_sre = (qry_sre/len(val_loader.dataset)) ** 0.5
    print("Test result: MAE/RMSE/NAE/SRE [{:5.2f},{:5.2f},{:5.2f},{:5.2f}]".format(qry_mae, qry_rmse,qry_nae,qry_sre))
    print("test runtime {}".format(runtime))

    return qry_mae, qry_rmse

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    main(args)