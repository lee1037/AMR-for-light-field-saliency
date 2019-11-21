# An Attentive Multi-Level Recurrent Network for Salient Object Detection on Light Field

## Overview of the framework
![Image text](pic/framework.jpg)

## Pre-train
Use `RGBdata` in [dataset.py](dataset.py) to load RGB saliency dataset for pre-training.

## Fine-tune
Use `LFdata` in [dataset.py](dataset.py) to load light field saliency dataset for pre-training.

## Models
The fine-turned models are in [parameters](parameters/).

## Test
Run `python test.py` to test. 

## Results
![Image text](pic/results.jpg)
Visual samples of the comparison results. Note that ‘Image’ is used to represent the scene, instead of the input of our AMR network.
Our AMR network (column (c)) can accurately and completely separate the salient object from the cluttered background in challenging
scenes, for example, when the background is cluttered or has similar appearance with the salient object.
