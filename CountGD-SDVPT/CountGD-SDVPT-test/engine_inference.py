# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import numpy as np
import math
import os
import sys
import io
import contextlib
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from util.utils import to_device
from util.visualizer import renorm
import torch
import torchvision.transforms.functional as F

import util.misc as utils
from util.misc import nested_tensor_from_tensor_list
from datasets_inference.cocogrounding_eval import CocoGroundingEvaluator
import time
from datasets_inference.panoptic_eval import PanopticEvaluator
from segment_anything import sam_model_registry, SamPredictor
from datasets_inference.transforms import RandomResize
from scipy.stats import bernoulli

from models_inference.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)



def get_sam(sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def numpy_2_cv2(np_img):
    if np.min(np_img) < 0:
        raise Exception("image min is less than 0. Img min: " + str(np.min(np_img)))
    if np.max(np_img) > 1:
        raise Exception("image max is greater than 1. Img max: " + str(np.max(np_img)))

    np_img = (np_img * 255).astype(np.uint8)
    # Need to somehow ensure image is in RGB format. Note this line shows up in SAM demo: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2_image = np.asarray(np_img)
    return cv2_image


def tt_norm_sam(predictor, pred_cnt, image, exemplars, size, points):
    e_cnt = 0
    avg_cnt = 0
    (h, w) = (size[0], size[1])
    xv, yv = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    image_cv = numpy_2_cv2(image)
    predictor.set_image(image_cv)

    for exemp in exemplars:
        in_exemp = (points[:, 0] * w > exemp[0]) * (points[:, 0] * w < exemp[2])
        in_exemp = (
            (in_exemp) * (points[:, 1] * h > exemp[1]) * (points[:, 1] * h < exemp[3])
        )
        # There are at least 2 points inside the exemplar [exemp].
        if np.sum(in_exemp) >= 2:
            print("refining tt norm with SAM")
            sam_mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=exemp[None, :],
                multimask_output=False,
            )
            x_mask = xv[sam_mask.squeeze()]
            y_mask = yv[sam_mask.squeeze()]
            mask_points = np.stack((x_mask, y_mask), axis=1)
            num_points_in_mask = 0
            for point in points:
                in_mask = False
                # Check if [point] lies inside the mask.
                for mask_point in mask_points:
                    if mask_point[0] == round(w * point[0]) and mask_point[1] == round(
                        h * point[1]
                    ):
                        in_mask = True
                        break
                if in_mask:
                    num_points_in_mask += 1

            # There is an exemplar mask with more than one detected instance.
            if num_points_in_mask >= 2:
                e_cnt += 1
            # Add to the average exemplar count using the SAM mask instead of the exemplar.
            avg_cnt += num_points_in_mask
        else:
            # Add to the average exemplar count using the exemplar.
            avg_cnt += np.sum(in_exemp)

    # If there are at least 2 exemplar masks with more than one detected instance, apply the TT-Norm.
    if e_cnt >= 2:
        avg_cnt = avg_cnt / exemplars.shape[0]
        print("Using TT-Norm")
        print("orig count: " + str(pred_cnt))
        pred_cnt = pred_cnt / avg_cnt
        print("new count: " + str(pred_cnt))

    return pred_cnt


def tt_norm(pred_cnt, exemplars, size, points):
    e_cnt = 0
    avg_cnt = 0
    (h, w) = (size[0], size[1])
    for exemp in exemplars:
        # Get number of points inside exemplar.
        in_exemp = (points[:, 0] * w > exemp[0]) * (points[:, 0] * w < exemp[2])
        in_exemp = (
            (in_exemp) * (points[:, 1] * h > exemp[1]) * (points[:, 1] * h < exemp[3])
        )
        if np.sum(in_exemp) >= 2:
            # There are at least 2 detected instances inside this exemplar.
            e_cnt += 1
        avg_cnt += np.sum(in_exemp)
    if e_cnt >= 2:
        avg_cnt = avg_cnt / exemplars.shape[0]
        # At least 2 of the exemplars contain 2 object instances.
        print("Using TT-Norm")
        print("orig count: " + str(pred_cnt))
        pred_cnt = pred_cnt / avg_cnt
        print("new count: " + str(pred_cnt))
    return pred_cnt


def crop(sample, crop_width, crop_height, overlap_width, overlap_height):
    (h, w) = sample.shape[1], sample.shape[2]

    samples_cropped = []
    start_y = 0
    end_y = crop_height
    start_x = 0
    end_x = crop_width
    boundaries_x = []
    boundaries_y = []

    while end_y < h:
        end_y = start_y + crop_height
        if end_y > h:
            # Shift up to increase overlap when hit bottom end of image.
            shift_up = end_y - h
            start_y = start_y - shift_up
            end_y = h
        boundaries_row_x = []
        boundaries_row_y = []
        while end_x < w:
            end_x = start_x + crop_width
            if end_x > w:
                # Shift left to increase overlap when hit right end of image.
                shift_left = end_x - w
                start_x = start_x - shift_left
                end_x = w
            samples_cropped.append(
                RandomResize([800], max_size=1333)(
                    sample[:, start_y:end_y, start_x:end_x].unsqueeze(0)
                )[0].squeeze()
            )
            boundaries_row_x.append((start_x, end_x))
            boundaries_row_y.append((start_y, end_y))
            start_x = start_x + crop_width - overlap_width
        boundaries_x.append(boundaries_row_x)
        boundaries_y.append(boundaries_row_y)

        start_x = 0
        end_x = crop_width
        start_y = start_y + crop_height - overlap_height

    return samples_cropped, boundaries_x, boundaries_y


def get_count_errs(
    model,
    args,
    samples,
    exemplars,
    outputs,
    box_threshold,
    text_threshold,
    targets,
    tokenized_captions,
    input_captions,
    predictor=None,
    train_text_embedding=None,
):
 
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    samples = samples.to_img_list()
    sizes = [target["size"] for target in targets]

    abs_errs = []
    for sample_ind in range(len(targets)):
        sample_logits = logits[sample_ind]
        sample_boxes = boxes[sample_ind]
        sample = samples[sample_ind]
        size = sizes[sample_ind]
        sample_exemplars = exemplars[sample_ind]

        sample_caption = input_captions[sample_ind]
        # Only use at most 3 exemplars for inference.
        assert sample_exemplars.shape[0] <= 3

        # Get the index ([end_idx]) where the special tokens end (the '.' separator indicates this for GroundingDINO).
        for token_ind in range(len(tokenized_captions["input_ids"][sample_ind])):
            idx = tokenized_captions["input_ids"][sample_ind][token_ind]
            if idx == 1012:
                end_idx = token_ind
                break

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        sample_logits = sample_logits[box_mask, :]
        sample_boxes = sample_boxes[box_mask, :]

        text_mask = (sample_logits[:, 1:end_idx] > text_threshold).sum(dim=-1) == (
            end_idx - 1
        )

        sample_logits = sample_logits[text_mask, :]
        sample_boxes = sample_boxes[text_mask, :]

        gt_count = targets[sample_ind]["labels_uncropped"].shape[0]
        pred_cnt = sample_logits.shape[0]

        # Predicted max # of objects.
        if args.crop and pred_cnt >= 220:
            # If crop image do not apply TT-Norm as double counting may occur on crop boundaries for a few (but not all) samples. In other words, cropping may (incorrectly) cause higher counts around boundaries, which may lead to false detection of self-similarity if TT-Norm is applied. Solution for now is to just disable TT-Norm when apply cropping.
            print("Detected high number of objects, cropping...")

            # Crop image.
            (h, w) = size[0], size[1]

            # Get crop size.
            obj_width = 0
            obj_height = 0
            for exemp in sample_exemplars:
                obj_width += exemp[2] - exemp[0]
                obj_height += exemp[3] - exemp[1]

            obj_width = int(obj_width / len(sample_exemplars))
            obj_height = int(obj_height / len(sample_exemplars))

            # Limit crop size to include approximately 16 objects assuming 16 << args.num_select.
            crop_width = 4 * obj_width
            crop_height = 4 * obj_height

            # Get overlap size. Ensures each object instance is fully seen by at least one crop window.
            overlap_width = round(1.25 * obj_width)
            overlap_height = round(1.25 * obj_height)

            samples_cropped, boundaries_x, boundaries_y = crop(
                sample, crop_width, crop_height, overlap_width, overlap_height
            )

            num_batches = int(np.ceil(len(samples_cropped) / 10))
            logits_cropped = []
            boxes_cropped = []

            pred_cnt = 0
            for batch_ind in range(num_batches):
                with torch.cuda.amp.autocast(enabled=args.amp):
                    # Use 'label' of 0 at inference since only input a single text prompt instead of all COCO classes (as do during training).
                    sample_subset = samples_cropped[
                        batch_ind * 10 : min((batch_ind + 1) * 10, len(samples_cropped))
                    ]
                    outputs_high_objects = model(
                        nested_tensor_from_tensor_list(sample_subset),
                        [sample_exemplars] * len(sample_subset),
                        [torch.tensor([0]).cuda() for _ in range(len(sample_subset))],
                        captions=[sample_caption] * len(sample_subset),
                        cropped=True,
                        orig_img=sample,
                        crop_width=crop_width,
                        crop_height=crop_height,
                        indexs=[1000 for i in range(len(sample_subset))],
                        mode='val',
                        raw_captions=[sample_caption] * len(sample_subset)
                    )
                logits_cropped.append(outputs_high_objects["pred_logits"].sigmoid())
                boxes_cropped.append(outputs_high_objects["pred_boxes"])

            logits_cropped = torch.cat(logits_cropped)
            boxes_cropped = torch.cat(boxes_cropped)

            for row_ind in range(len(boundaries_x)):
                for col_ind in range(len(boundaries_x[0])):
                    crop_ind = row_ind * len(boundaries_x[0]) + col_ind
                    sample_logits_cropped = logits_cropped[crop_ind]
                    sample_boxes_cropped = boxes_cropped[crop_ind]
                    start_x, end_x = (
                        boundaries_x[row_ind][col_ind][0],
                        boundaries_x[row_ind][col_ind][1],
                    )
                    start_y, end_y = (
                        boundaries_y[row_ind][col_ind][0],
                        boundaries_y[row_ind][col_ind][1],
                    )

                    # Get the index ([end_idx]) where the special tokens end (the '.' separator indicates this for GroundingDINO).
                    for token_ind in range(
                        len(tokenized_captions["input_ids"][sample_ind])
                    ):
                        idx = tokenized_captions["input_ids"][sample_ind][token_ind]
                        if idx == 1012:
                            end_idx = token_ind
                            break

                    box_mask = sample_logits_cropped.max(dim=-1).values > box_threshold
                    sample_logits_cropped = sample_logits_cropped[box_mask, :]
                    sample_boxes_cropped = sample_boxes_cropped[box_mask, :]

                    text_mask = (
                        sample_logits_cropped[:, 1:end_idx] > text_threshold
                    ).sum(dim=-1) == (end_idx - 1)
                    sample_logits_cropped = sample_logits_cropped[text_mask, :]
                    sample_boxes_cropped = sample_boxes_cropped[text_mask, :]

                    pred_crop_cnt = 0
                    for box in sample_boxes_cropped:
                        # Refer to region definitions in [cropping.pdf] to understand A, B, C, D, E, F, G, H, & I
                        (x, y) = crop_width * box[0], crop_height * box[1]
                        (transformed_x, transformed_y) = (x + start_x).item(), (
                            y + start_y
                        ).item()

                        # Case 1) central block:
                        if start_x > 0 and start_y > 0 and end_x < w and end_y < h:
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]

                            # Check regions A, D, G:
                            if x < (end_x_left - start_x):
                                # A:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # D:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 2
                                # G:
                                elif y >= crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 4

                            # Check regions B, E, H:
                            elif x >= (end_x_left - start_x) and x < crop_width - (
                                end_x - start_x_right
                            ):
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                elif y >= crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 2
                            # Check regions C, F, I:
                            elif x >= crop_width - (end_x - start_x_right):
                                # C:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # F:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 2
                                # I:
                                else:
                                    pred_crop_cnt += 1 / 4

                        # Case 2) left top corner block:
                        elif start_x == 0 and start_y == 0:
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]

                            # Check regions E, H:
                            if x < crop_width - (end_x - start_x_right):
                                # E:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                else:
                                    pred_crop_cnt += 1 / 2

                            # Check regions F, I:
                            else:
                                # F:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 2
                                # I:
                                else:
                                    pred_crop_cnt += 1 / 4

                        # Case 3) right top corner block
                        elif end_x == w and start_y == 0:
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]

                            # Check regions D, G:
                            if x < (end_x_left - start_x):
                                # D:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 2
                                # G:
                                else:
                                    pred_crop_cnt += 1 / 4

                            # Check regions E, H:
                            else:
                                # E:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                else:
                                    pred_crop_cnt += 1 / 2

                        # Case 4) left bottom corner block:
                        elif start_x == 0 and end_y == h:
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            # Check regions B, E:
                            if x < crop_width - (end_x - start_x_right):
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                else:
                                    pred_crop_cnt += 1 / 1
                            # Check regions C, F:
                            else:
                                # C:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # F:
                                else:
                                    pred_crop_cnt += 1 / 2

                        # Case 5) right bottom corner block:
                        elif end_x == w and end_y == h:
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            # Check regions A, D:
                            if x < (end_x_left - start_x):
                                # A:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # D:
                                else:
                                    pred_crop_cnt += 1 / 2
                            # Check regions B, E:
                            else:
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                else:
                                    pred_crop_cnt += 1 / 1

                        # Case 6) top central block:
                        elif start_y == 0 and (start_x > 0 and end_x < w):
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]
                            # Check regions D, G:
                            if x < (end_x_left - start_x):
                                # D:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 2
                                # G:
                                else:
                                    pred_crop_cnt += 1 / 4
                            # Check regions E, H:
                            elif x >= (end_x_left - start_x) and x < crop_width - (
                                end_x - start_x_right
                            ):
                                # E:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                else:
                                    pred_crop_cnt += 1 / 2
                            # Check regions F, I:
                            else:
                                # F:
                                if y < crop_height - (end_y - start_y_down):
                                    pred_crop_cnt += 1 / 2

                                # I:
                                else:
                                    pred_crop_cnt += 1 / 4

                        # Case 7) bottom central block:
                        elif end_y == h and (start_x > 0 and end_x < w):
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            # Check regions A, D:
                            if x < (end_x_left - start_x):
                                # A:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # D:
                                else:
                                    pred_crop_cnt += 1 / 2

                            # Check regions B, E:
                            elif x >= (end_x_left - start_x) and x < crop_width - (
                                end_x - start_x_right
                            ):
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                else:
                                    pred_crop_cnt += 1 / 1
                            # Check regions C, F:
                            else:
                                # C:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # F:
                                else:
                                    pred_crop_cnt += 1 / 2

                        # Case 8) left central block:
                        elif start_x == 0 and (start_y > 0 and end_y < h):
                            start_x_right = boundaries_x[row_ind][col_ind + 1][0]
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]
                            # Check regions B, E, H:
                            if x < crop_width - (end_x - start_x_right):
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                else:
                                    pred_crop_cnt += 1 / 2
                            # Check regions C, F, I:
                            else:
                                # C:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # F:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 2
                                # I:
                                else:
                                    pred_crop_cnt += 1 / 4

                        # Case 9) right central block:
                        elif end_x == w and (start_y > 0 and end_y < h):
                            end_y_up = boundaries_y[row_ind - 1][col_ind][1]
                            start_y_down = boundaries_y[row_ind + 1][col_ind][0]
                            end_x_left = boundaries_x[row_ind][col_ind - 1][1]
                            # Check regions A, D, G:
                            if x < (end_x_left - start_x):
                                # A:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 4
                                # D:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 2
                                # G:
                                else:
                                    pred_crop_cnt += 1 / 4
                            # Check regions B, E, H:
                            else:
                                # B:
                                if y < (end_y_up - start_y):
                                    pred_crop_cnt += 1 / 2
                                # E:
                                elif y >= (end_y_up - start_y) and y < crop_height - (
                                    end_y - start_y_down
                                ):
                                    pred_crop_cnt += 1 / 1
                                # H:
                                else:
                                    pred_crop_cnt += 1 / 2
                        # Raise exception if no case is met.
                        else:
                            raise Exception(
                                "Detected box is not in any of the provided blocks!"
                            )

                    pred_cnt += pred_crop_cnt


        if args.simple_crop and pred_cnt >= 220:
            # If crop image do not apply TT-Norm as double counting may occur on crop boundaries for a few (but not all) samples. In other words, cropping may (incorrectly) cause higher counts around boundaries, which may lead to false detection of self-similarity if TT-Norm is applied. Solution for now is to just disable TT-Norm when apply cropping.
            print("Detected high number of objects, cropping...")

            # When using the simple crop, just divide image into 4 regions and sum the predicted count for each region.
            (h, w) = size[0], size[1]
            sample_top_left = F.resize(sample[:, : (h // 2), : (w // 2)], (h, w))
            sample_bot_left = F.resize(sample[:, (h // 2) : h, : (w // 2)], (h, w))
            sample_top_right = F.resize(sample[:, : (h // 2), (w // 2) : w], (h, w))
            sample_bot_right = F.resize(sample[:, (h // 2) : h, (w // 2) : w], (h, w))
            pred_cnt = 0
            with torch.cuda.amp.autocast(enabled=args.amp):
                # Use 'label' of 0 at inference since only input a single text prompt instead of all COCO classes.
                outputs_high_objects = model(
                    nested_tensor_from_tensor_list(
                        [
                            sample_top_left,
                            sample_bot_left,
                            sample_top_right,
                            sample_bot_right,
                        ]
                    ),
                    [sample_exemplars] * 4,
                    [torch.tensor([0]).cuda() for _ in range(4)],
                    captions=input_captions * 4,
                    cropped=True,
                    orig_img=sample,
                    indexs=[1000,1000,1000,1000], #random integer, only the training phase of TGPR needs this variable
                    mode='val',
                    raw_captions=input_captions * 4,
                    train_text_embedding=train_text_embedding
                )
            logits_cropped = outputs_high_objects["pred_logits"].sigmoid()
            boxes_cropped = outputs_high_objects["pred_boxes"]

            for crop_ind in range(4):
                sample_logits_cropped = logits_cropped[crop_ind]
                sample_boxes_cropped = boxes_cropped[crop_ind]
                sample_cropped = [
                    sample_top_left,
                    sample_bot_left,
                    sample_top_right,
                    sample_bot_right,
                ][crop_ind]
                size_cropped = (h, w)
                # Get the index ([end_idx]) where the special tokens end (the '.' separator indicates this for GroundingDINO).
                for token_ind in range(
                    len(tokenized_captions["input_ids"][sample_ind])
                ):
                    idx = tokenized_captions["input_ids"][sample_ind][token_ind]
                    if idx == 1012:
                        end_idx = token_ind
                        break

                box_mask = sample_logits_cropped.max(dim=-1).values > box_threshold
   
                sample_logits_cropped = sample_logits_cropped[box_mask, :]
                sample_boxes_cropped = sample_boxes_cropped[box_mask, :]

                text_mask = (sample_logits_cropped[:, 1:end_idx] > text_threshold).sum(
                    dim=-1
                ) == (end_idx - 1)
                sample_logits_cropped = sample_logits_cropped[text_mask, :]
                sample_boxes_cropped = sample_boxes_cropped[text_mask, :]
                pred_cnt += sample_logits_cropped.shape[0]

 
        elif args.sam_tt_norm:
            pred_cnt = tt_norm_sam(
                predictor,
                pred_cnt,
                renorm(sample.cpu()).permute(1, 2, 0).numpy(),
                sample_exemplars.cpu().numpy(),
                size.cpu().numpy(),
                sample_boxes[:, :2].cpu().numpy(),
            )
        elif args.exemp_tt_norm:
            pred_cnt = tt_norm(
                pred_cnt,
                sample_exemplars.cpu().numpy(),
                size.cpu().numpy(),
                sample_boxes[:, :2].cpu().numpy(),
            )

        print("Pred Count: " + str(pred_cnt) + ", GT Count: " + str(gt_count))

        abs_errs.append(np.abs(gt_count - pred_cnt))

    return abs_errs,gt_count


@torch.no_grad()
def evaluate(
    model,
    model_without_ddp,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):
    model.eval()
    criterion.eval()

    if args.sam_tt_norm:
        predictor = get_sam(sam_checkpoint=args.sam_model_path)
    else:
        predictor = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item["name"] for item in category_dict]
    else:
        cat_list = args.val_label_list


    abs_errs = []
    norm_abs_errs_nae = []
    norm_abs_errs_sre = []

    train_name_list=['alcohol bottle .', 'baguette roll .', 'ball .', 'banana .', 'bead .', 'bee .', 'birthday candle .', 'biscuit .', 'boat .', 'bottle .', 'bowl .', 'box .', 'bread roll .', 'brick .', 'buffalo .', 'bun .', 'calamari ring .', 'can .', 'candle .', 'cap .', 'car .', 'cartridge .', 'cassette .', 'cement bag .', 'cereal .', 'chewing gum piece .', 'chopstick .', 'clam .', 'coffee bean .', 'coin .', 'cotton ball .', 'cow .', 'crane .', 'crayon .', 'croissant .', 'crow .', 'cup .', 'cupcake .', 'cupcake holder .', 'fish .', 'gemstone .', 'go game piece .', 'goat .', 'goldfish snack .', 'goose .', 'ice cream .', 'ice cream cone .', 'instant noodle .', 'jade stone .', 'jeans .', 'kidney bean .', 'kitchen towel .', 'lighter .', 'lipstick .', 'm&m piece .', 'macaron .', 'match .', 'meat skewer .', 'mini blind .', 'mosaic tile .', 'naan bread .', 'nail .', 'nut .', 'onion ring .', 'orange .', 'pearl .', 'pen .', 'pencil .', 'penguin .', 'pepper .', 'person .', 'pigeon .', 'plate .', 'polka dot tile .', 'potato .', 'rice bag .', 'roof tile .', 'screw .', 'shoe .', 'spoon .', 'spring roll .', 'stair .', 'stapler pin .', 'straw .', 'supermarket shelf .', 'swan .', 'tomato .', 'watermelon .', 'window .', 'zebra .']
    train_text_token = model.tokenizer(train_name_list, padding="longest", return_tensors="pt").to(device)
    with torch.no_grad():
        (text_self_attention_masks,position_ids,cate_to_token_mask_list,) = generate_masks_with_special_tokens_and_transfer_map(train_text_token, model.specical_tokens, model.tokenizer)
        tokenized_for_encoder = {k: v for k, v in train_text_token.items() if k != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks
        tokenized_for_encoder["position_ids"] = position_ids
        bert_output = model.bert(**tokenized_for_encoder)  # bs, 195, 768
        #using only last token for similarity calculation
        train_text_embedding = bert_output["last_hidden_state"][torch.arange(len(train_name_list)),tokenized_for_encoder["position_ids"].argmax(dim=-1)] # 取句号这个token、bs, d_model


    for samples, targets in metric_logger.log_every(
        data_loader, 10, header, logger=logger
    ):
        samples = samples.to(device)
        
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        exemplars = [t["exemplars"][: args.num_exemplars].to(device) for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [cat_list[target["labels"][0]] + " ." for target in targets]

        if args.no_text:
            input_captions = [" ." for target in targets]

        print("input_captions: " + str(input_captions))
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(
                samples,
                exemplars,
                [torch.tensor([0]).to(device) for _ in targets],
                captions=input_captions,
                indexs=[20 for target in targets], # random integer, only the training phase of TGPR needs this variable
                mode='val', # As long as mode! = 'train' OR 'trian2', the SDPE for inference is used
                raw_captions=input_captions,
                train_text_embedding=train_text_embedding
            )
            tokenized_captions = outputs["token"]
            errs,gt_count = get_count_errs(
                model,
                args,
                samples,
                exemplars,
                outputs,
                args.box_threshold,
                args.text_threshold,
                targets,
                tokenized_captions,
                input_captions,
                predictor=predictor,
                train_text_embedding=train_text_embedding,
    
            )
            abs_errs += errs
            norm_abs_errs_nae += [i/gt_count if gt_count!=0 else 0 for i in errs ]
            norm_abs_errs_sre += [(i**2)/gt_count if gt_count!=0 else 0 for i in errs ]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res["boxes"]
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1
                )

                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())

                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break
    count_mae = sum(abs_errs) / len(abs_errs)
    count_rmse = (np.array(abs_errs) ** 2).mean() ** (1 / 2)
    count_nae = sum(norm_abs_errs_nae) / len(norm_abs_errs_nae)
    count_sre = (sum(norm_abs_errs_sre) / len(norm_abs_errs_sre)) ** (1 / 2)
    print("# of Images Tested: " + str(len(abs_errs)))
    print("MAE: " + str(count_mae) + ", RMSE: " + str(count_rmse))
    print("NAE: " + str(count_nae) + ", SRE: " + str(count_sre))

    

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, "results-{}.pkl".format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    
    with contextlib.redirect_stdout(io.StringIO()):
        # Suppress print output from unused [GroundingDINO] functions.
        if coco_evaluator is not None:
            coco_evaluator.accumulate()
            coco_evaluator.summarize()
    
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]

    return count_mae, stats, coco_evaluator
