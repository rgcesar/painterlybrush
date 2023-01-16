# painterlybrush



A python implementation of a painting algorithm described in a paper "Painterly Rendering with Curved Brush Strokes of Multiple Sizes" by Aaron Hertzmann.

## Summary of the algorithm



The algorithm produces painterly renderings of images by using layers of different brush sizes. In each layer, the brush paints a curved stroke. The color and location of the stroke comes from the largest color difference between the reference image and the rendering canvas. The strokes are curved by using the image gradient to produce spline strokes.

## Painting styles



Hertzman's paper gives parameters for 4 painting styles: Imperssionist, Expressionist, Colorist Wash, and Pointillist. These have been included in the examples.

## Usage



Rendering an image using one of the provided styles ```impressionist, expressionist, colorist_wash, pointillist```:
```bash
$ python3 ./painterlybrush.py -i input_image -o output_image.png --impressionist
Style Parameters: T=100 , R=[8, 4, 2], fc=1, fs=0.5, alpha=1, fg=1, minl=4, maxl=16
Rendering image...
Done! Saved to ./output_image.png

```

## Results

![Bird](../assets/painterlybird.png?raw=true | width=200)
![OBird](../assets/bird.jpg | width=200)



## Requirements

 * OpenCV
 * NumPy

 ## Pending improvements

 

Most of the style parameters have been implemented. Color jitter remains to be implemented for randomly adding jitter to the hue, saturation, value, red, green, or blue components.
