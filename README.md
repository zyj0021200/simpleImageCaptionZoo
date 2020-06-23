# simple Image Caption Zoo
This is a PyTorch implementation (more like a **tutorial**) of Image Caption Models. However, this project does not win with the variety of STOA models covered. Its main purpose is to introduce the common operations and tricks involved in the reproduction process of Image Caption models. I tried to make the code as simple as possible, while the comments are as detailed as possible. Considering I am not a native English speaker, some of my comment statements may not be accurate enough. The project has covered the following content:

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
- [ ] Support loading pre-trained [bottom-up features](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)

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

- Download the original COCO14 [Training](http://images.cocodataset.org/zips/train2014.zip) and [Validation](http://images.cocodataset.org/zips/val2014.zip) files and put them in the folders in 'Datasets/MSCOCO/2014/', I have created some empty folders for the corresponding locations, you just need to unzip files to these locations. 

- Download [Karpathy's split files](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This file contains captions and image splits information for Flickr8K/30K and COCO14. You need to put the **dataset_coco.json** to **/Datasets/MSCOCO/2014/**
- Run the **Datasets_json_modification.py** to generate modified annotations for training and testing. Currently supports nltk/PTBTokenizer/Karpathy tokenization for raw captions. You could check the original and modified annotations examples using **show_original_COCO14_annotations_example** and **show_modified_jsons** function.
- **Notice1:** different word-tokenization optimizations have slightly effect on the model performance except for the length and content of caption_vocab. Karpathy has already tokenized the raw captions for flickr8k/30k and COCO14, while he didn't tokenize on the COCO17 dataset and he treated the possessives of all nouns specially. For example the karpathy tokenization result of sentence "a dog sits on a man's leg" is ['a','dog','sits','on','a','mans','leg']，while nltk and PTB may generate ['a','dog','sits','on','a','man'," 's ",'leg']. Considering that the official COCO evaluation code uses PTBtokenizer to tokenize the sentences and calculate the scores, we default that tokenizer. You could easily change that in the ArgumentParser in **Datasets_json_modification.py**
- **Notice2:** You only need to perform similar operations on Flickr datasets by dumping the **dataset_flickr8k.json** and **dataset_flickr30k.json** to **/Datasets/Flickr/8K(30K)**, and dump  the image files to **/Datasets/Flickr/8K(30K)/images**. However, since Karpathy didn't preprocess on COCO17. You should simply put the original COCO17 files to corresponding locations.

## About Models

- Currently I only reproduce the NIC model and a simplified Bottom-Up & Top-Down model. I didn't use the pretrained bottom-up features. For simplicity, I use the CNN feature map instead. All models inherit from the Engine class, so you could easily build your own fancy models.

## Get Started

- I will update the pretrained checkpoints soon, but I really suggest that you train the model from scratch. Remember to pre-train the model with **XELoss** before **SCST** training. For more operational instructions, please refer to the comments in **Main.py**.

