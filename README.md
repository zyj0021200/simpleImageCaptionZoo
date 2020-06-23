# simpleImageCaptionZoo
This is a PyTorch implementation of Image Caption Models. However, this project does not win with the variety of STOA models covered. Its main purpose is to introduce the common operations and tricks involved in the reproduction process of Image Caption models. I tried to make the code as simple as possible, while the comments are as detailed as possible. Considering I am not a native English speaker, some of my comment statements may not be accurate enough. The project has covered the following content:

- Support training and testing for **four datasets** (Flickr8K/30K,COCO14/17)
- **Beam_search** strategy when evaluating
- **Scheduled Sampling** & learning rate Decay during training process
- **Self-Critical Sequence Training** is included
- Support **attention map visualization**

To increase the readability of the code and help you get started as quickly as possible. 

- You only need to read from the first line of **Main.py** to grasp the entire project
- Add **detailed comments** to the input and output of all functions
- Ensure that the code for each model is concentrated in **one python file**

TODO:

- [ ] comments for every single line in the code
- [ ] tensorboard visualization
