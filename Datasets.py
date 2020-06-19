import torch
import os
import copy
import nltk
import torch.utils.data as tdata
from DatasetClass import CaptionData
import string
from PIL import Image

#------------------------------Dataset utils-------------------------------#
def get_img_path(img_root,img_filename,dataset_name,split=None):
    img_path = None
    if dataset_name in ['Flickr8K','Flickr30K']:
        img_path = os.path.join(img_root, img_filename)
    elif dataset_name == 'COCO14':
        if 'train' in img_filename.lower():
            img_path = os.path.join(img_root, 'train2014', img_filename)
        else:
            img_path = os.path.join(img_root, 'val2014', img_filename)
    elif dataset_name == 'COCO17':
        img_path = os.path.join(img_root,split+'2017',img_filename)
    return img_path

#----------------------------Datasets---------------------------------------#
#-(anns_keys)
class CaptionTrainDataset(tdata.Dataset):
    def __init__(self,img_root,cap_ann_path,vocab,img_transform=None,dataset_name=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.vocab = vocab
        self.ids = list(self.capdata.anns.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        ann_id = self.ids[index]
        img_id = self.capdata.anns[ann_id]['image_id']
        img_filename = self.capdata.anns[ann_id]['file_name']
        img_path = get_img_path(img_root=self.img_root,img_filename=img_filename,dataset_name=self.dataset_name,split='train')
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = None
        if self.img_transform is not None:
            transformed_img = self.img_transform(original_img)
            img_tensor = copy.deepcopy(transformed_img)
        tokens = self.capdata.anns[ann_id]['tokens']
        caption = []
        caption.append(self.vocab('<sta>'))
        caption.extend(self.vocab(token) for token in tokens)
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return img_id,img_tensor,target

    def __len__(self):
        return len(self.ids)

#-(imgs_keys)
class CaptionTrainSCSTDataset(tdata.Dataset):
    def __init__(self,img_root,cap_ann_path,img_transform=None,dataset_name=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.ids = list(self.capdata.imgs.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_filename = self.capdata.imgs[img_id]['file_name']
        img_path = get_img_path(img_root=self.img_root,img_filename=img_filename,dataset_name=self.dataset_name,split='train')
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = None
        if self.img_transform is not None:
            transformed_img = self.img_transform(original_img)
            img_tensor = copy.deepcopy(transformed_img)

        img_entry = self.capdata.imgs[img_id]   #{'file_name': 'COCO_val2014_000000522418.jpg', 'id': 522418, 'sentids': [681330, 686718, 688839, 693159, 693204], 'sentences': [{'tokens': ['a', 'woman', 'wearing',......
        gt_captions = []
        for sent in img_entry['sentences']:
            token_list = sent['tokens']
            cap = ' '.join(token_list)
            gt_captions.append(cap)
        img_gt_captions = {img_id:gt_captions}

        return img_id,img_tensor,img_gt_captions

    def __len__(self):
        return len(self.ids)

#-(imgs_keys)
class CaptionEvalDataset(tdata.Dataset):
    def __init__(self,img_root,cap_ann_path,img_transform=None,dataset_name=None,eval_split=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.ids = list(self.capdata.imgs.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name
        self.eval_split = eval_split

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_filename = self.capdata.imgs[img_id]['file_name']
        img_path = get_img_path(img_root=self.img_root,img_filename=img_filename,dataset_name=self.dataset_name,split=self.eval_split)
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = None
        if self.img_transform is not None:
            transformed_img = self.img_transform(original_img)
            img_tensor = copy.deepcopy(transformed_img)
        return img_id,img_tensor

    def __len__(self):
        return len(self.ids)

#-------------Dataloader_collate_fn------------------#
def COCOCaptionTrain_collate_fn(data):
    data.sort(key=lambda x:len(x[2]),reverse=True)
    img_ids,images,captions = zip(*data)
    images = torch.stack(images,0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return img_ids,images,targets,lengths

def COCOCaptionTrainSCST_collate_fn(data):
    img_ids,images,img_gts = zip(*data)
    images = torch.stack(images,0)
    gts = {}
    for gt in img_gts:
        gts.update(gt)
    return img_ids,images,gts