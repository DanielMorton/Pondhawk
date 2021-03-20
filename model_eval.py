import cv2
import os
import numpy as np
import tensorflow as tf
import configargparse


def get_test_images(image_dir, ext='.jpg', test=False):
    files = [f"{image_dir}/{f}" for f in os.listdir(image_dir) if ext in f.lower()]
    files.sort()
    if test:
        files = [f for i, f in enumerate(files) if not i % 8]
    return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]


def get_poses(image_dir, test=True):
    poses_arr = np.load(os.path.join(image_dir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    if test:
        poses, bds = poses[::8], bds[::8]
    return poses, bds


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        help='Image Directory Path')
    parser.add_argument("--pred_dir", help='Rendering Directory')
    parser.add_argument("--test", help='Only use test images',  action='store_true')
    args = parser.parse_args()

    images = get_test_images(args.image_dir, test=args.test)
    pred_images = get_test_images(args.pred_dir, ext='.png')
    if len(images) != len(pred_images):
        print("Pred count does not match image count")
        return

    pred_shape = pred_images[0].shape
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (pred_shape[1], pred_shape[0]))

    psnr = [tf.image.psnr(img, pred,  255).numpy() for img, pred in zip(images, pred_images)]
    ms_ssim = [tf.image.ssim_multiscale(img, pred, max_val=255).numpy() for img, pred in zip(images, pred_images)]

    print(f"PSNR for the image set is {np.mean(psnr)} +- {np.std(psnr)}.")
    print(f"MS-SSIM for the image set in {np.mean(ms_ssim)} +- {np.std(ms_ssim)}.")


if __name__ == '__main__':
    main()
