# Quanti.us (Detection Code)

This repository contains the representative implementation of the Fully Convolutional Regression Network used for the detection task in [1]. Please note the following directory structure.

|- code  
|- annotation  
|- data  
|- dependency [this folder is currently absent, please create it and add MatConvNet library here]  

The annotation/9_gt_expert.csv has one expert's annotation for the entire dataset (all 48 frames). The training session uses this ground truth. The other CSV files in the annotation directory correspond to the annotations obtained from six experts in total (used primarily for testing/evaluation puprose). The MATLAB script code/demo_train_and_test.m demonstrates the full process.

## Dependency
[MatConvNet: CNNs for MATLAB] (http://www.vlfeat.org/matconvnet/)

## Citation

Hughes, Alex J., Joseph D. Mornin, Sujoy K. Biswas, Lauren E. Beck, David P. Bauer, Arjun Raj, Simone Bianco, and Zev J. Gartner. ["Quanti.us: a tool for rapid, flexible, crowd-based annotation of images"](https://www.nature.com/articles/s41592-018-0069-0/) Nature methods 15, no. 8 (2018): 587

