# Pondhawk

Ancillary code associated with stress testing NeRF. This presupposes Tensorflow and COLMAP are already installed.

## Running COLMAP

In order to compute the camera poses necessary for NeRF, run 

```./run_colmap.sh --d IMAGE_DIR```

The standard format for colmap image directory is a main ```IMAGE_DIR``` with an ```images``` folder containing the images. To simplify creation of such a
directory, run 

```./make_dir.sh -d IMAGE_DIR```

After COLMAP is run, it's output will be in the ```IMAGE_DIR/sparse``` directory.

## Evaluating NeRF

As NeRF runs it produces predictions for the test set every 50K iterations. To compute PSNR and MS-SSIM scores for these test images simply run

```python model_eval.py --image_dir IMAGE_DIR --pred_dir PRED_DIR --test```

Remove the ```--test``` tag to compute the scores for the full image set. Otherwise, only every eighth images (NeRFs standard test set) will be used.

Example images are provided. The command to run the evaluation for them is

```python model_eval.py --image_dir ./images --pred_dir ./test_050000 --test```


## NeRF Keras Model

The code to construct the NeRF Keras model is also included. the ```layers``` package contains all the custom layers needed for model construction, and
```model.py``` contains the functions for making the coarse, fine, and full Keras models.
