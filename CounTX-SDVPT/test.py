import os
import argparse
import datetime
import json
import numpy as np
import time
from pathlib import Path
from PIL import Image

import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import util.misc as misc
from util.FSC147 import TTensor
from models_counting_network import CountingNetwork
import open_clip
from scipy import stats

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Testing Open-world Text-specified Object Counting Network"
    )

    parser.add_argument(
        "--data_split",
        default="test",
        help="data split of FSC-147 to test",
    )

    parser.add_argument(
        "--output_dir",
        default="./test",
        help="path where to save test log",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for testing")

    parser.add_argument(
        "--resume",
        default="./counting_network.pth",
        help="file name for model checkpoint to use for testing",
    )

    parser.add_argument("--num_workers", default=1, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_false",
        help="pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )


    parser.add_argument(
        "--img_dir",
        default="/home/eamon/CLIP-Count-main/data/FSC/images_384_VarV2",
        help="directory containing images from FSC-147",
    )

    parser.add_argument(
        "--gt_dir",
        default="/home/eamon/CLIP-Count-main/data/FSC/gt_density_map_adaptive_384_VarV2",
        help="directory containing ground truth binary dot annotation maps",
    )

    parser.add_argument(
        "--class_file",
        default="/home/eamon/CLIP-Count-main/data/FSC/FSC_147/ImageClasses_FSC_147.txt",
        help="name of file with FSC-147 image class names",
    )

    parser.add_argument(
        "--FSC147_anno_file",
        default="/home/eamon/CLIP-Count-main/data/FSC/FSC_147/annotation_FSC_147_384.json",
        help="name of file with FSC-147 annotations",
    )

    parser.add_argument(
        "--FSC147_D_anno_file",
        default="./FSC-147-D.json",
        help="name of file with FSC-147-D",
    )

    parser.add_argument(
        "--data_split_file",
        default="/home/eamon/CLIP-Count-main/data/FSC/FSC_147/Train_Test_Val_FSC_147.json",
        help="name of file with train, val, test splits of FSC-147",
    )
    return parser


open_clip_vit_b_16_preprocess = transforms.Compose(
    [
        transforms.Resize(
            size=224,
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias="warn",
        ),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class TestData(Dataset):
    def __init__(self, args):

        self.img_dir = args.img_dir

        with open(args.data_split_file) as f:
            data_split = json.load(f)
        self.img = data_split[args.data_split]

        with open(args.FSC147_anno_file) as f:
            fsc147_annotations = json.load(f)
        self.fsc147_annotations = fsc147_annotations

        with open(args.FSC147_D_anno_file) as f:
            fsc147_d_annotations = json.load(f)
        self.fsc147_d_annotations = fsc147_d_annotations

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        fsc147_anno = self.fsc147_annotations[im_id]
        fsc147_d_anno = self.fsc147_d_annotations[im_id]
        text = self.clip_tokenizer(fsc147_d_anno["text_description"]).squeeze(-2)

        dots = np.array(fsc147_anno["points"])

        image = Image.open("{}/{}".format(self.img_dir, im_id))
        image.load()
        W, H = image.size

        # This resizing step exists for consistency with CounTR's data resizing step.
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        image = transforms.Resize((new_H, new_W))(image)
        image = TTensor(image)

        return image, dots, text,1000


def main(args):

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # Force PyTorch to be deterministic for reproducibility. See https://pytorch.org/docs/stable/notes/randomness.html.
    
    # torch.use_deterministic_algorithms(True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    dataset_test = TestData(args)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Initialize the model.
    model = CountingNetwork()

    model.to(device)

    train_name_list=['the alcohol bottles', 'the baguette rolls', 'the balls', 'the balls of yarn', 'the bananas', 'the beads', 'the bees', 'the birthday candles', 'the biscuits', 'the boats', 'the bottles', 'the bowls', 'the boxes', 'the bread rolls', 'the bricks', 'the buffaloes', 'the buns', 'the calamari rings', 'the candles', 'the cans', 'the caps', 'the cars', 'the cartridges', 'the cassettes', 'the cement bags', 'the cereal boxes', 'the cereals', 'the chewing gum pieces', 'the chopsticks', 'the cinnamon buns', 'the cinnamon rolls', 'the clams', 'the coffee beans', 'the coins', 'the cones in the cupcake tray', 'the cotton balls', 'the cows', 'the cranes', 'the crayons', 'the croissants', 'the crows', 'the cupcake holders', 'the cupcake holders in the cupcake tray', 'the cupcake holders of the cupcake tray', 'the cupcake slots', 'the cupcake slots in the blue cupcake tray', 'the cupcake slots in the cupcake tray', 'the cupcake slots in the cupcake trays', 'the cupcake slots in the metal cupcake tray', 'the cupcake spots in the tray', 'the cupcakes', 'the cupcakes in or around the cupcake tray', 'the cupcakes in the cupcake tray', 'the cupcakes in the cupcake trays', 'the cupcakes on the cupcake tray', 'the cupcakes on the dress', 'the cups', 'the cups of ice cream', 'the empty cupcake holders in the cupcake tray', 'the empty cupcake holders in the metal cupcake tray', 'the empty cupcake holders in the metal tray', 'the empty cupcake holders in the two cupcake trays', 'the empty cupcake slots in the cupcake tray', 'the empty cupcake slots in the tray', 'the empty slots in the cupcake tray', 'the fishes', 'the geese', 'the gemstones', 'the go game pieces', 'the goats', 'the goldfish snack pieces', 'the heart-shaped cupcake holders', 'the individual shoes', 'the jade stones', 'the jeans', 'the kidney beans', 'the kitchen towels', 'the lighters', 'the lipsticks', 'the m&m pieces', 'the macarons', 'the matches', 'the meat skewers', 'the mini blinds', 'the mosaic tiles', 'the naan bread', 'the nails', 'the nuts', 'the onion rings', 'the oranges', 'the packages of ice cream', 'the packages of instant noodles', 'the pearls', 'the pencils', 'the penguins', 'the pens', 'the people', 'the peppers', 'the pigeons', 'the plants in the cupcake tray', 'the plates', 'the poker balls', 'the polka dot tiles', 'the potatoes', 'the rice bags', 'the roof tiles', 'the screws', 'the shoes', 'the slots in the cupcake tray', 'the spoons', 'the spring rolls', 'the stacked cupcakes', 'the stairs', 'the stapler pins', 'the straws', 'the supermarket shelves', 'the swans', 'the tennis balls', 'the the empty cupcake slots in the cupcake tray', 'the tomatoes', 'the watermelons', 'the windows', 'the zebras']
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")
    train_text = clip_tokenizer(train_name_list).squeeze(-2).to(device, non_blocking=True)
    train_text_tokens=model.foward_txt_encoder(train_text).unsqueeze(0)

    misc.load_model_FSC(args=args, model_without_ddp=model)

    print(f"Start testing.")
    start_time = time.time()

    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Testing (" + args.data_split + ")"
    print_freq = 20

    test_mae = 0
    test_rmse = 0
    test_nae = 0
    test_sre = 0

    for data_iter_step, (samples, gt_dots, text_descriptions,class_ids) in enumerate(
        metric_logger.log_every(data_loader_test, print_freq, header)
    ):

        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()
        text_descriptions = text_descriptions.to(device, non_blocking=True)
        class_ids=class_ids.to('cpu').numpy().tolist() 
        _, _, h, w = samples.shape
        # Apply sliding window density map averaging technique used in CounTR.
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:

                (output,outs_vpt_class,outs_vpt_weighted) = model(
                    open_clip_vit_b_16_preprocess(
                        samples[:, :, :, start : start + 384]
                    ),
                    text_descriptions,
                    'val', # As long as mode! = 'train' OR 'trian2', it is SDPE 
                    class_ids,
                    train_text_tokens
                )
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0 : prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1 : 384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start : prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1 : w])

                density_map = (
                    density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                )

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

        pred_cnt = torch.sum(density_map / 60).item()

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)

        
        test_mae += cnt_err
        test_rmse += cnt_err**2
        test_nae+=cnt_err/gt_cnt
        test_sre+=(cnt_err**2)/gt_cnt
        print(
            f"{data_iter_step}/{len(data_loader_test)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  AE: {cnt_err},  SE: {cnt_err ** 2},  NAE: {cnt_err/gt_cnt},  SRE: {(cnt_err ** 2)/gt_cnt} "
        )

    print("Averaged stats:", metric_logger)

    log_stats = {
        "MAE": test_mae / (len(data_loader_test)),
        "RMSE": (test_rmse / (len(data_loader_test))) ** 0.5,
        "NAE": test_nae / (len(data_loader_test)),
        "SRE": (test_sre / (len(data_loader_test))) ** 0.5
    }

    print(
        "Test MAE: {:5.2f}, Test RMSE: {:5.2f} ,Test NAE: {:5.2f}, Test SRE: {:5.2f} ".format(
            test_mae / (len(data_loader_test)),
            (test_rmse / (len(data_loader_test))) ** 0.5,
            test_nae / (len(data_loader_test)),
            (test_sre / (len(data_loader_test))) ** 0.5,
        )
    )

    with open(
        os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

 

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
