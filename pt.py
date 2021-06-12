import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from scipy.misc import imresize

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = 'cuda'


def load_image(imfile):
    img = Image.open(imfile).resize((1024, 1024))
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def float_image_resize(img, shape, interp='bilinear'):
    missing_channel = False
    if len(img.shape) == 2:
        missing_channel = True
        img = img[..., None]
    layers = []
    img = img.transpose(2, 0, 1)
    for l in img:
        l = imresize(l, shape, interp=interp, mode='F')
        layers.append(l)
    if missing_channel:
        return np.stack(layers, axis=-1)[..., 0]
    else:
        return np.stack(layers, axis=-1)


def visualize_corrs(img1, img2, corrs, mask=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = imresize(img1, scale1)
    img2 = imresize(img2, scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = .5
    alpha = 1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )
    plt.scatter(xs, ys)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def raft_flow_a_to_b(model, imfile1, imfile2):
    image1_shape = Image.open(imfile1).size
    image2_shape = Image.open(imfile2).size
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    # viz(image1, flow_up)
    x, y = np.meshgrid(np.linspace(0, 1024 - 1, 1024),
                       np.linspace(0, 1024 - 1, 1024))
    base_grid = np.stack([x, y], axis=-1)
    flow_up_np = flow_up[0].cpu().permute(1, 2, 0).numpy() + base_grid
    scale_x = image2_shape[0] / 1024
    scale_y = image2_shape[1] / 1024
    flow_up_np[..., 0] *= scale_x
    flow_up_np[..., 1] *= scale_y
    flow_up_np = float_image_resize(flow_up_np, image1_shape[::-1])
    # return flow_up_np
    # import IPython
    # IPython.embed()
    # assert 0

    # img_a = imageio.imread(imfile1)
    # img_b = imageio.imread(imfile2)
    # x, y = np.meshgrid(np.linspace(0, 1024 - 1, 1024),
    #                np.linspace(0, 1024 - 1, 1024))
    # base_grid = np.stack([x, y], axis=-1)
    # flow_up_np = flow_up[0].cpu().permute(1,2,0).numpy() + base_grid
    # warped = cv2.remap(img_b, flow_up_np[..., 0].astype(np.float32), flow_up_np[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return flow_up_np, None
    import IPython
    IPython.embed()
    assert 0


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    test_img_dir = '/ubc/cs/research/kmyi/jw221/imw-2020-test/st_pauls_cathedral'
    images = []
    for _, _, files in os.walk(test_img_dir):
        for f in files:
            if not f.endswith('.npy'):
                images.append(f)
        images = sorted(images)
        break
    print(images)

    corr_dir = './temp/spc'
    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))

        # images = sorted(images)
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
            # imfile1, imfile2 = imfile2, imfile1
        counter = 0
        for img_name1 in images:
            for img_name2 in images:
                if img_name1 <= img_name2:
                    continue
                counter += 1
                print(counter)
                # continue
                corr_path = os.path.join(corr_dir, f'{img_name1}->{img_name2}_corrs.npy')
                # img_name1 = imfile1
                # img_name2 = imfile2
                imfile1 = os.path.join(test_img_dir, img_name1)
                imfile2 = os.path.join(test_img_dir, img_name2)
                flow12, warped1 = raft_flow_a_to_b(model, imfile1, imfile2)
                flow21, warped2 = raft_flow_a_to_b(model, imfile2, imfile1)

                cycle = cv2.remap(flow12, flow21[..., 0].astype(np.float32), flow21[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR)

                im2_h, im2_w, _ = flow21.shape
                x, y = np.meshgrid(np.linspace(0, im2_w - 1, im2_w),
                                   np.linspace(0, im2_h - 1, im2_h))
                base_grid = np.stack([x, y], axis=-1)
                PIXEL_THRESHOLD = 5
                mask_cycle = np.linalg.norm(cycle - base_grid, axis=2) < PIXEL_THRESHOLD
                print(mask_cycle.sum())
                # mask_cycle[mask_cycle==True]=False
                index = np.random.choice(mask_cycle.sum(), min(2048, mask_cycle.sum()), replace=False)
                pixel_ind = np.array(np.where(mask_cycle)).T[index]
                start = base_grid[pixel_ind[:, 0], pixel_ind[:, 1]]
                end = flow21[pixel_ind[:, 0], pixel_ind[:, 1]]
                corrs = np.concatenate([end, start], axis=1)

                # img_a = imageio.imread(imfile1)
                # img_b = imageio.imread(imfile2)

                np.save(corr_path, corrs)
                # import IPython
                # IPython.embed()
                # assert 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
