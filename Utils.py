import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from pycocotools.coco import COCO
import json
import torch
import skimage.transform
import torchvision.transforms as transforms
from DatasetClass import CaptionData
from Models.NIC_Model import NIC_Captioner
from Models.BUTDSpatial_Model import BUTDSpatial_Captioner
from Models.AoASpatial_Model import AoASpatial_Captioner
from Build_Vocab import build_vocab
import torch.utils.data as tdata
from Datasets import CaptionTrainDataset,CaptionEvalDataset,CaptionTrainSCSTDataset,COCOCaptionTrain_collate_fn,COCOCaptionTrainSCST_collate_fn

#----------------------data utils--------------------------#
def parse_data_config(path,base_dir):
    """Parses the data configuration file"""
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        if value.find('/') != -1:
            value = base_dir + value
        options[key.strip()] = value.strip()
    return options

def get_caption_vocab(args,opt):
    caption_vocab_path = opt['caption_vocab_path']
    caption_json_path = opt['train_caption_path']
    if os.path.exists(caption_vocab_path):
        caption_vocab_file = open(caption_vocab_path, 'rb')
        caption_vocab = pickle.load(caption_vocab_file)
        print('Caption Vocab for dataset:%s loaded complete.' % args.dataset)
    else:
        caption_vocab = build_vocab(json=caption_json_path, threshold=5)
        with open(caption_vocab_path, 'wb') as f:
            pickle.dump(caption_vocab, f)
        print("Total vocabulary size: %d" % len(caption_vocab))
        print("Saved the vocabulary wrapper to '%s'" % caption_vocab_path)
    return caption_vocab

def get_train_dataloader(args,opt,caption_vocab):
    train_img_transform = get_transform(resized_img_size=args.img_size, enhancement=['RandomHorizontalFlip'])
    train_dataset = CaptionTrainDataset(img_root=opt['image_root'], cap_ann_path=opt['train_caption_path'],
                                        vocab=caption_vocab, img_transform=train_img_transform,
                                        dataset_name=args.dataset)
    train_dataloader = tdata.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=4, collate_fn=COCOCaptionTrain_collate_fn)
    print('Finish initialing train_dataloader.')
    return train_dataloader

def get_eval_dataloader(args,opt,eval_split):
    eval_img_transform = get_transform(resized_img_size=args.img_size, enhancement=[])
    if eval_split == 'val':cap_ann_path = opt['val_caption_path']
    else:cap_ann_path = opt['test_caption_path']
    eval_dataset = CaptionEvalDataset(img_root=opt['image_root'], cap_ann_path=cap_ann_path,
                                      img_transform=eval_img_transform, dataset_name=args.dataset, eval_split=eval_split)
    if args.eval_beam_size != -1:
        eval_batch_size = 1
    else:
        eval_batch_size = args.eval_batch_size
    eval_dataloader = tdata.DataLoader(dataset=eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    print('Finish initialing eval_dataloader.')
    return eval_dataloader

def get_scst_train_dataloader(args,opt):
    train_img_transform = get_transform(resized_img_size=args.img_size, enhancement=['RandomHorizontalFlip'])
    scst_train_dataset = CaptionTrainSCSTDataset(img_root=opt['image_root'], cap_ann_path=opt['train_caption_path'],
                                                 img_transform=train_img_transform, dataset_name=args.dataset)
    scst_train_dataloader = tdata.DataLoader(dataset=scst_train_dataset, batch_size=args.scst_train_batch_size,
                                             shuffle=True, num_workers=4, collate_fn=COCOCaptionTrainSCST_collate_fn)
    print('Finish initialing scst_train_dataloader.')
    return scst_train_dataloader

def get_sample_image_info(img_root,img_filename):
    gts = {}
    if 'COCO' and '2014' in img_filename:
        datname,splitname,imgidname = img_filename.split('_')   #'COCO','val2014','0XXX0243832.jpg'
        split = splitname[:-4]
        img_id = imgidname.split('.')[0]
        img_id = int(img_id)
        data_root = img_root
        img_root = os.path.join(img_root,splitname) #'./Datasets/MSCOCO/2014/val2014/'
        img = Image.open(os.path.join(img_root,img_filename))
        original_img = img.convert('RGB')
        plt.imshow(original_img)
        plt.title('%s Image:\n%s' % (datname, img_filename), fontsize=20)
        plt.axis('off')
        plt.show()
        annFile = os.path.join(data_root,'annotations','captions_%s2014.json' % split)
        coco = COCO(annFile)
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        print('Ground truth captions:')
        coco.showAnns(anns)
        gts = {img_id:[entry for entry in anns]}
    elif 'Flickr' in img_root:
        img = Image.open(os.path.join(img_root,img_filename))
        dat_name = 'Flickr'
        if '8K' in img_root:sub_name = '8K'
        else:sub_name = '30K'
        fk_data_root = img_root[:-7]    #./Datasets/Flickr/8K/
        fktrain = CaptionData(annotation_file=fk_data_root+'modified_annotations/captions_train.json')
        fkval = CaptionData(annotation_file=fk_data_root+'modified_annotations/captions_val.json')
        original_img = img.convert('RGB')
        plt.imshow(original_img)
        plt.title('%s Image:\n%s' % (dat_name+sub_name, img_filename), fontsize=20)
        plt.axis('off')
        plt.show()
        if fktrain.filenameToImgid.__contains__(img_filename):
            img_id = fktrain.filenameToImgid[img_filename]
            anns = fktrain.imgToAnns[img_id]
        else:
            img_id = fkval.filenameToImgid[img_filename]
            anns = fkval.imgToAnns[img_id]
        print('Ground truth captions:')
        for entry in anns:
            print(entry['caption'])
        gts = {img_id:[entry for entry in anns]}
    else:
        img = Image.open(os.path.join(img_root,img_filename))
        original_img = img.convert('RGB')
        plt.imshow(original_img)
        plt.title('Sample Image:\n%s' % (img_filename), fontsize=20)
        plt.axis('off')
        plt.show()
    return img,gts

#------------------------model utils------------------------#
def model_construction(model_settings_json,caption_vocab,device):
    settings = json.load(open(model_settings_json,'r'))
    model = None
    if settings['model_type'] == 'NIC':
        model = NIC_Captioner(
            embed_dim=settings['embed_dim'],
            hidden_dim=settings['hidden_dim'],
            vocab_size=len(caption_vocab),
            device=device
        )
    elif settings['model_type'] == 'BUTDSpatial':
        model = BUTDSpatial_Captioner(
            encoded_img_size=settings['enc_img_size'],
            atten_dim=settings['atten_dim'],
            embed_dim=settings['embed_dim'],
            hidden_dim=settings['hidden_dim'],
            vocab_size=len(caption_vocab),
            device=device
        )
    elif settings['model_type'] == 'AoASpatial':
        model = AoASpatial_Captioner(
            encoded_img_size=settings['enc_img_size'],
            embed_dim=settings['embed_dim'],
            hidden_dim=settings['hidden_dim'],
            vocab_size=len(caption_vocab),
            device=device
        )
    return model,settings

def get_transform(resized_img_size=224,enhancement=[]):
    transform_instructions = [transforms.Resize((resized_img_size, resized_img_size), interpolation=Image.LANCZOS)]
    if 'RandomHorizontalFlip' in enhancement:
        transform_instructions.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'RandomVerticalFlip' in enhancement:
        transform_instructions.append(transforms.RandomVerticalFlip(p=0.5))
    transform_instructions.append(transforms.ToTensor())
    transform_instructions.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    img_transform = transforms.Compose(transforms=transform_instructions)
    return img_transform

def init_SGD_optimizer(params, lr, momentum=0.9, weight_decay=1e-5):
    return torch.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
def init_Adam_optimizer(params, lr):
    return torch.optim.Adam(params=params, lr=lr)

def init_optimizer(optimizer_type,params,learning_rate):
    optimizer = None
    if len(params)>0:
        if optimizer_type == 'Adam':
            optimizer = init_Adam_optimizer(params=params,lr=learning_rate)
        elif optimizer_type == 'SGD':
            optimizer = init_SGD_optimizer(params=params,lr=learning_rate)
    return optimizer

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

#----------------------sample utils------------------#
def visualize_att(image, alphas, caption, img_size=448, smooth=True, STAflag=False):
    """
    Visualizes caption with weights at every word.
    :param image: (3,H,W)
    :param alphas: weights (L-1,h*w)
    :param caption: [<sta>,'a','man',...,'.',<end>]
    :param smooth: smooth weights?
    """
    image = image.resize([img_size, img_size], Image.LANCZOS)

    for t in range(len(caption)):
        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=img_size/current_alpha.shape[0], sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [img_size, img_size])
        if t == 0 and STAflag:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
        plt.show()