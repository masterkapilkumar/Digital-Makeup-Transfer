# Digital Makeup Transfer
It can transfer the makeup from an example image to any other base image. It requires the images to have a straight face.

It uses extended Active Shape Model (ASM) to obtain face control points, denaulay triangulation and Affine Transformation for establishing correspondance between images, WLS filter for layer decomposition (face structure layer and skin detail layer), gradient editing method for preserving texture of lips while transferring the lip makeup.

## Dependencies
- python 2.x
- opencv for python
- PyStasm
- numpy, scipy, matplotlib

## How to use

In terminal, change the working directory to "src" folder and run the following command:
``` 
python makeup_transfer.py path_to_base_image path_to_example_image
```

This will create 4 output images:
```
    result.jpg - final output image
    morphed.jpg - example image with morphed face w.r.t. base image
    base_points.jpg - base image with face control points
    example_points.jpg - example image with face control points
```