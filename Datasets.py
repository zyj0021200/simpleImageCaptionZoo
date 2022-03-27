import torch
import os
import copy
import numpy as np
import pickle
import torch.utils.data as tdata
from ClassRepository.DatasetClass import CaptionData
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
    def __init__(self,img_root,cap_ann_path,vocab,img_transform=None,dataset_name=None,supp_infos=[],supp_dir=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.vocab = vocab
        self.ids = list(self.capdata.anns.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name
        self.supp_infos = supp_infos
        self.supp_dir = supp_dir

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
        #--------for supplementary informations--------------#
        supp_info_data = {}
        if 'fixed_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'fixed_bu_feat/%s.npz' % (str(img_id))))['feat']  #(36,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'fixed_bu_bbox/%s.npy' % (str(img_id))))  #(36,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})
        elif 'adaptive_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'adaptive_bu_feat/%s.npz' % (str(img_id))))['feat']   #(10~100,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'adaptive_bu_bbox/%s.npy' % (str(img_id))))   #(10~100,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})

        return img_id,img_tensor,target,supp_info_data

    def __len__(self):
        return len(self.ids)

#-(imgs_keys)
class CaptionTrainSCSTDataset(tdata.Dataset):
    def __init__(self,img_root,cap_ann_path,img_transform=None,dataset_name=None,supp_infos=[],supp_dir=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.ids = list(self.capdata.imgs.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name
        self.supp_infos = supp_infos
        self.supp_dir = supp_dir

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

        #--------for supplementary informations--------------#
        supp_info_data = {}
        if 'fixed_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'fixed_bu_feat/%s.npz' % (str(img_id))))['feat']  #(36,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'fixed_bu_bbox/%s.npy' % (str(img_id))))  #(36,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})
        elif 'adaptive_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'adaptive_bu_feat/%s.npz' % (str(img_id))))['feat']   #(10~100,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'adaptive_bu_bbox/%s.npy' % (str(img_id))))   #(10~100,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})

        return img_id,img_tensor,img_gt_captions,supp_info_data

    def __len__(self):
        return len(self.ids)

#-(imgs_keys)
class CaptionEvalDataset(tdata.Dataset):
    def __init__(self,img_root,cap_ann_path,img_transform=None,dataset_name=None,eval_split=None,supp_infos=[],supp_dir=None):
        self.img_root = img_root
        self.capdata = CaptionData(annotation_file=cap_ann_path)
        self.ids = list(self.capdata.imgs.keys())
        self.img_transform = img_transform
        self.dataset_name = dataset_name
        self.eval_split = eval_split
        self.supp_infos = supp_infos
        self.supp_dir = supp_dir

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_filename = self.capdata.imgs[img_id]['file_name']
        img_path = get_img_path(img_root=self.img_root,img_filename=img_filename,dataset_name=self.dataset_name,split=self.eval_split)
        original_img = Image.open(img_path).convert('RGB')
        img_tensor = None
        if self.img_transform is not None:
            transformed_img = self.img_transform(original_img)
            img_tensor = copy.deepcopy(transformed_img)

        #--------for supplementary informations--------------#
        supp_info_data = {}
        if 'fixed_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'fixed_bu_feat/%s.npz' % (str(img_id))))['feat']  #(36,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'fixed_bu_bbox/%s.npy' % (str(img_id))))  #(36,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})
        elif 'adaptive_bu_feat' in self.supp_infos:
            bu_feat = np.load(os.path.join(self.supp_dir, 'adaptive_bu_feat/%s.npz' % (str(img_id))))['feat']   #(10~100,2048)
            bu_bbox = np.load(os.path.join(self.supp_dir, 'adaptive_bu_bbox/%s.npy' % (str(img_id))))   #(10~100,4)
            supp_info_data.update({'bu_feat':bu_feat,'bu_bbox':bu_bbox})

        return img_id,img_tensor,supp_info_data

    def __len__(self):
        return len(self.ids)

#-------------Dataloader_collate_fn------------------#
def COCOCaptionTrain_collate_fn(data):
    data.sort(key=lambda x:len(x[2]),reverse=True)
    img_ids,img_tensors,captions,supp_info_datas = zip(*data)
    img_tensors = torch.stack(img_tensors,0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return img_ids,img_tensors,targets,lengths,supp_info_datas

def COCOCaptionTrainSCST_collate_fn(data):
    img_ids,img_tensors,img_gts,supp_info_datas = zip(*data)
    img_tensors = torch.stack(img_tensors,0)
    gts = {}
    for gt in img_gts:
        gts.update(gt)
    return img_ids,img_tensors,gts,supp_info_datas

def COCOCaptionEval_collate_fn(data):
    img_ids,img_tensors,supp_info_datas = zip(*data)
    img_tensors = torch.stack(img_tensors,0)
    return img_ids,img_tensors,supp_info_datas

if __name__ == '__main__':
    img_root = './Datasets/MSCOCO/2014/'
    train_cap_path = './Datasets/MSCOCO/2014/modified_annotations/captions_train.json'
    eval_cap_path = './Datasets/MSCOCO/2014/modified_annotations/captions_test.json'
    vocab = pickle.load(open('./Data/MSCOCO/2014/caption_vocab.pkl','rb'))
    import torchvision.transforms as transforms
    img_transform = transforms.Compose([
        transforms.Resize((224,224),interpolation=Image.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset_name = 'COCO14'
    supp_dir = './Data/MSCOCO/2014/'
    train_dataset = CaptionTrainDataset(
        img_root=img_root,
        cap_ann_path=train_cap_path,
        vocab=vocab,
        img_transform=img_transform,
        dataset_name=dataset_name,
        supp_infos=[],
        supp_dir=supp_dir
    )
    train_dataloader = tdata.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=COCOCaptionTrain_collate_fn
    )
    eval_dataset = CaptionEvalDataset(
        img_root=img_root,
        cap_ann_path=eval_cap_path,
        img_transform=img_transform,
        dataset_name=dataset_name,
        eval_split='test',
        supp_infos=['adaptive_bu_feat'],
        supp_dir=supp_dir
    )
    eval_dataloader = tdata.DataLoader(
        dataset=eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=COCOCaptionEval_collate_fn
    )
    scst_train_dataset = CaptionTrainSCSTDataset(
        img_root=img_root,
        cap_ann_path=train_cap_path,
        img_transform=img_transform,
        dataset_name=dataset_name,
        supp_infos=['adaptive_bu_feat'],
        supp_dir=supp_dir
    )
    scst_train_dataloader = tdata.DataLoader(
        dataset=scst_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=COCOCaptionTrainSCST_collate_fn
    )
    import time
    import tqdm
    t0 = time.time()
    train_data_it = iter(train_dataloader)
    print(next(train_data_it))
    eval_data_it = iter(eval_dataloader)
    print(next(eval_data_it))
    scst_data_it = iter(scst_train_dataloader)
    print(next(scst_data_it))
    for (_,_,_) in tqdm.tqdm(eval_dataloader):
        pass

    t1 = time.time()
    print('iteration time: %.2fs' % (t1-t0))
