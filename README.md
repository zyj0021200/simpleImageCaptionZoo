# simple Image Caption Zoo (Updating)

### Most recent update
- support pretrained faster-rcnn bottom-up-features
- support BUTD and AoA model
- add code comments for **Data_json_modification.py**

## Introduction
This is a PyTorch implementation of Image Caption Models. However, this project does not win with the variety of STOA models covered. Its main purpose is to introduce the common operations and tricks involved in the reproduction process of Image Caption models. I tried to make the code as simple as possible, while the code comments are as detailed as possible. Considering I am not a native English speaker, some of my comment statements may not be accurate enough. The project has covered the following content:

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
- [x] Support loading pre-trained [bottom-up features](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip)

Many thanks to the authors and their codes for the following references:

- https://github.com/ruotianluo/self-critical.pytorch
- https://github.com/fawazsammani/show-edit-tell/
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- https://github.com/poojahira/image-captioning-bottom-up-top-down
- https://github.com/husthuaan/AoANet
- ......

## Results obtained (Updating)

| Model         | XE/ w/o Beam Seaerch | XE/ w/  beam search | SCST/ w/o beam search | SCST/ w/ beam search |
| ------------- | -------------------- | ------------------- | --------------------- | -------------------- |
| NIC           | 93.1                 | 96.5                | 103.8                 | 104.3                |
| BUTDSpatial   | 97.3                 | 103.0               | 110.2                 | 110.5                |
| BUTDDetection | 106.1                | 111.4               | -                     | -                    |


Results reported on COCO14 Karpathy test split.

## Data Preparation(COCO14 as example)

- Download the original COCO14 [Training](http://images.cocodataset.org/zips/train2014.zip) and [Validation](http://images.cocodataset.org/zips/val2014.zip) files and put them in the folders in 'Datasets/MSCOCO/2014/', I have created some empty folders for the corresponding locations, you just need to unzip files to these locations. 

- Download [Karpathy's split files](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This file contains captions and image splits information for Flickr8K/30K and COCO14. You need to put the **dataset_coco.json** to **/Datasets/MSCOCO/2014/**
- Run the **Datasets_json_modification.py** to generate modified annotations for training and testing. Currently supports nltk/PTBTokenizer/Karpathy tokenization for raw captions. You could check the original and modified annotations examples using **show_original_annotation_jsonfiles** and **show_modified_annotation_jsonfiles** function.
- **Notice1:** different word-tokenization optimizations have slightly effect on the model performance except for the length and content of caption_vocab. Karpathy has already tokenized the raw captions for flickr8k/30k and COCO14, while he didn't tokenize on the COCO17 dataset and he treated the possessives of all nouns specially. For example the karpathy tokenization result of sentence "a dog sits on a man's leg" is ['a','dog','sits','on','a','mans','leg']，while nltk and PTB may generate ['a','dog','sits','on','a','man'," 's ",'leg']. Considering that the official COCO evaluation code uses PTBtokenizer to tokenize the sentences and calculate the scores, we default that tokenizer. You could easily change that in the **ArgumentParser** in **Datasets_json_modification.py**
- **Notice2:** You only need to perform similar operations on Flickr datasets by dumping the **dataset_flickr8k.json** and **dataset_flickr30k.json** to **/Datasets/Flickr/8K(30K)**, and dump  the original image files to **/Datasets/Flickr/8K(30K)/images**. However, since Karpathy didn't preprocess on COCO17. You should simply put the original COCO17 files to corresponding locations.  Considering that COCO17 dataset only rearranges COCO14 to a certain extent, so training for COCO17 is **optional**.

## About Models

- Currently I only reproduce **3 typical types** of models: the **traditional NIC** model from CVPR2015; the fundamental **Bottom-Up & Top-Down model** form CVPR2018; the STOA **Attention on Attention model** from ICCV2019. You could decide whether to use the pretrained **faster-rcnn bottom-up features** or the **CNN feature map** as the bottom_up features for the last two models. 
- All models inherit from the Engine class, so you could easily **build your own fancy models**. The model you write needs to have a similar **output result format** as the model I provided. During the training phase, your model needs output a pack_padded prediction. During the evaluating phase, you need to output a **(bsize,max_seq)** torch tensor to represent the generated captions. If your model needs to **input additional data** during the training phase, it is recommended that you put these data in the **opt['data_dir']** directory and overwrite the **training_epoch** and **eval_captions_json_generation** function.

## About Configs

- All the **dir/path/root** information for different datasets are stored in **./Configs/Datasets/DATASETNAME.json**, you could modify them if necessary.
- All the model settings are stored in **./Configs/Models/MODELNAME.json**. Note that I saved some recommended learning rate in the json file. You could choose to use the preset settings or not by adjusting the **use_preset** option in **Main.py** 
- You could dump your own json file in the folder for your models.

## About SCST

- Self-Critical Sequence Training requires the **tf-idf** results of your training corpus. So you need to run the **Cider_idf_preprocess.py** first to generate the corpus file **DATASETNAME-train.p**. 
- It takes longer to train the models under SCST, please be patient.

## Get Started

- I will update the pretrained checkpoints soon, but I really suggest that you train the model from scratch. Remember to pre-train the model with **XELoss** before **SCST** training. For more operational instructions, please refer to the comments in **Main.py**.

## Visualization Examples

<div align=center><img src="https://github.com/zyj0021200/simpleImageCaptionZoo/blob/master/images/bbox_atten_BUTD.png" width="600" height="450" /></div>
<div align=center><img src="https://github.com/zyj0021200/simpleImageCaptionZoo/blob/master/images/spatial_atten_BUTD.png" width="600" height="450" /></div>


