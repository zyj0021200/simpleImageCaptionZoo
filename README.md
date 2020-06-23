# simple Image Caption Zoo
This is a PyTorch implementation(more like a tutorial) of Image Caption Models. However, this project does not win with the variety of STOA models covered. Its main purpose is to introduce the common operations and tricks involved in the reproduction process of Image Caption models. I tried to make the code as simple as possible, while the comments are as detailed as possible. Considering I am not a native English speaker, some of my comment statements may not be accurate enough. The project has covered the following content:

- Support training and testing for **four datasets** (Flickr8K/30K,COCO14/17)
- **Beam_search** strategy when evaluating
- **Scheduled Sampling** & learning rate Decay during training process
- **Self-Critical Sequence Training** is included
- Support **attention map visualization**

To increase the readability of the code and help you get started as quickly as possible. 

- You only need to read from the first line of **Main.py** to grasp the entire project.
- Add **detailed comments** to the input and output of all functions (TODO).
- Ensure that the code for each model is concentrated in **one python file**.

TODO:

- [ ] Comments for every single line in the code
- [ ] Tensorboard visualization
- [ ] Support more models

Many thanks to the authors and their codes for the following references:

- https://github.com/ruotianluo/self-critical.pytorch
- https://github.com/fawazsammani/show-edit-tell/
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/poojahira/image-captioning-bottom-up-top-down
- ......

## Results obtained

![results](D:\5 Programme\PyCharmProjects\results.png)

Results reported on COCO14 Karpathy test split.

## Data Preparation(COCO14 as example)

- Download the original COCO14 [Training](http://images.cocodataset.org/zips/train2014.zip) and [Validation](http://images.cocodataset.org/zips/val2014.zip) files and put them in the folders in 'Datasets/MSCOCO/2014/', I have created some empty folders for the corresponding locations, you just need to unzip files to these locations

- Download [Karpathy's split files](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This file contains captions and split imformation for Flickr8K/30K and COCO14. You need to put the **dataset_coco.json** to **/Datasets/MSCOCO/2014/**
- 
