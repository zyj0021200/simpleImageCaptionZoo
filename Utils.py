import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import torch.nn.functional as Fun
from pycocotools.coco import COCO
import json
import torch
import skimage.transform
import torchvision.transforms as transforms
from ClassRepository.DatasetClass import CaptionData
from Models.NIC_Model import NIC_Captioner
from Models.BUTD_Model import BUTDSpatial_Captioner,BUTDDetection_Captioner
from Models.AoA_Model import AoASpatial_Captioner,AoADetection_Captioner
import torch.utils.data as tdata
import torch.nn as nn
import numpy as np
from cider.pyciderevalcap.ciderD.ciderD import CiderD
from Datasets import CaptionTrainDataset,CaptionEvalDataset,CaptionTrainSCSTDataset,COCOCaptionTrain_collate_fn,COCOCaptionTrainSCST_collate_fn,COCOCaptionEval_collate_fn

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

def get_train_dataloader(args,opt,caption_vocab,supp_infos=[]):
    train_img_transform = get_transform(resized_img_size=args.img_size, enhancement=['RandomHorizontalFlip'])
    train_dataset = CaptionTrainDataset(
        img_root=opt['image_root'],
        cap_ann_path=opt['train_caption_path'],
        vocab=caption_vocab,
        img_transform=train_img_transform,
        dataset_name=args.dataset,
        supp_infos=supp_infos,
        supp_dir=opt['data_dir']
    )
    train_dataloader = tdata.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=COCOCaptionTrain_collate_fn
    )
    print('Finish initialing train_dataloader.')
    return train_dataloader

def get_eval_dataloader(args,opt,eval_split,use_beam=False,supp_infos=[]):
    eval_img_transform = get_transform(resized_img_size=args.img_size, enhancement=[])
    if eval_split == 'val':cap_ann_path = opt['val_caption_path']
    else:cap_ann_path = opt['test_caption_path']
    eval_dataset = CaptionEvalDataset(
        img_root=opt['image_root'],
        cap_ann_path=cap_ann_path,
        img_transform=eval_img_transform,
        dataset_name=args.dataset,
        eval_split=eval_split,
        supp_infos=supp_infos,
        supp_dir=opt['data_dir']
    )
    if use_beam:
        eval_batch_size = 1
    else:
        eval_batch_size = args.eval_batch_size
    eval_dataloader = tdata.DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=COCOCaptionEval_collate_fn
    )
    print('Finish initialing eval_dataloader.')
    return eval_dataloader

def get_scst_train_dataloader(args,opt,supp_infos=[]):
    train_img_transform = get_transform(resized_img_size=args.img_size, enhancement=['RandomHorizontalFlip'])
    scst_train_dataset = CaptionTrainSCSTDataset(
        img_root=opt['image_root'],
        cap_ann_path=opt['train_caption_path'],
        img_transform=train_img_transform,
        dataset_name=args.dataset,
        supp_infos=supp_infos,
        supp_dir=opt['data_dir']
    )
    scst_train_dataloader = tdata.DataLoader(
        dataset=scst_train_dataset,
        batch_size=args.scst_train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=COCOCaptionTrainSCST_collate_fn
    )
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
    elif settings['model_type'] == 'BUTDDetection':
        model = BUTDDetection_Captioner(
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
    elif settings['model_type'] == 'AoADetection':
        model = AoADetection_Captioner(
            embed_dim=settings['embed_dim'],
            hidden_dim=settings['hidden_dim'],
            vocab_size=len(caption_vocab),
            device=device
        )
    return model,settings

#-----------------------training utils--------------------------#
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
    return torch.optim.Adam(params=params, lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0)

def init_optimizer(optimizer_type,params,learning_rate):
    optimizer = None
    if len(params)>0:
        if optimizer_type == 'Adam':
            optimizer = init_Adam_optimizer(params=params,lr=learning_rate)
        elif optimizer_type == 'SGD':
            optimizer = init_SGD_optimizer(params=params,lr=learning_rate)
    return optimizer

def set_lr(optimizer, lr_list):
    for i,group in enumerate(optimizer.param_groups):
        group['lr'] = lr_list[i]

def get_lr(optimizer):
    lr_list = []
    for group in optimizer.param_groups:
        lr_list.append(group['lr'])
    return lr_list

def clip_gradient(optimizer, grad_clip=0.1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target):
        '''
        Perform label_smoothing loss calculation on pack_padded_predictions and targets
        :param input: pack_padded_sequence (total_num_words_in_batch,vocab_size)
        :param target: pack_padded_sequence (total_num_words_in_batch)
        :return: loss: floatTensor
        '''
        # assert x.size(1) == self.size
        input = Fun.log_softmax(input,dim=-1)
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1)).sum() / input.size(0)

# SCST code mainly adapted from
# https://github.com/ruotianluo/self-critical.pytorch and https://github.com/fawazsammani/show-edit-tell/

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sample_logprobs, seq, reward):
        '''
        Compute the reward for this batch
        :param sample_logprobs: (bsize,max_len)
        :param seq: (bsize,max_len)
        :param reward: (bsize,max_len)
        :return: output: torch.FloatTensor
        '''

        sample_logprobs = sample_logprobs.view(-1)  # (batch_size * max_len)
        reward = reward.view(-1)
        # set mask elements for all <end> tokens to 0
        mask = (seq > 0).float()  # (batch_size, max_len)
        # account for the <end> token in the mask. We do this by shifting the mask one timestep ahead
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)

        if not mask.is_contiguous():
            mask = mask.contiguous()

        mask = mask.view(-1)
        output = - sample_logprobs * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

def get_self_critical_reward(gen_result, greedy_res, ground_truth, img_ids, caption_vocab, dataset_name, cider_weight=1):
    '''
    :param gen_result: (batch_size, max_len)
    :param greedy_res: (batch_size, max_len)
    :param ground_truth:
    :param img_ids:
    :param caption_vocab:
    :param dataset_name:
    :param cider_weight:
    :return:
    '''

    batch_size = gen_result.size(0)
    res = []
    gen_result = gen_result.data.cpu().numpy()  # (batch_size, max_len)
    greedy_res = greedy_res.data.cpu().numpy()  # (batch_size, max_len)

    for image_idx in range(batch_size):
        sampled_ids = gen_result[image_idx]
        for endidx in range(len(sampled_ids)-1,-1,-1):
            if sampled_ids[endidx] != 0:
                break
        sampled_ids = sampled_ids[:endidx+1]
        sampled_caption = []
        for word_id in sampled_ids:
            word = caption_vocab.ix2word[word_id]
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        res.append({'image_id': img_ids[image_idx], 'caption': [sentence]})

    for image_idx in range(batch_size):
        sampled_ids = greedy_res[image_idx]
        sampled_caption = []
        for word_id in sampled_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':break
            sampled_caption.append(word)
        sentence = ' '.join(sampled_caption)
        res.append({'image_id': img_ids[image_idx], 'caption': [sentence]})

    CiderD_scorer = CiderD(df='%s-train' % dataset_name)
    _, cider_scores = CiderD_scorer.compute_score(ground_truth, res)

    scores = cider_weight * cider_scores
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # gen_result.shape[1] = max_len
    rewards = torch.from_numpy(rewards).float()

    return rewards

#----------------------sample utils------------------#
def visualize_att(image, alphas, caption, img_size=448, smooth=True):
    """
    Visualizes caption with weights at every word.
    :param image: (W,H)
    :param alphas: weights (L-1,h,w)
    :param caption: [<sta>,'a','man',...,'.',<end>]
    :param smooth: smooth weights?
    """
    image = image.resize([img_size, img_size], Image.LANCZOS)

    for t in range(len(caption)):
        plt.subplot(np.ceil(len(caption) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=img_size/current_alpha.shape[0], sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [img_size, img_size])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

def visualize_att_bboxes(image, alphas, bboxes, caption, img_size=448):
    """
    Visualizes caption with weights at every word.
    :param image: (W,H)
    :param alphas: weights (L-1,36)
    :param bboxes: (36,4) numpy.ndarry
    :param caption: [<sta>,'a','man',...,'.',<end>]
    :param smooth: smooth weights?
    """
    W,H = image.size
    image = image.resize([img_size, img_size], Image.LANCZOS)

    for t in range(len(caption)):
        plt.subplot(np.ceil(len(caption) / 5.), 5, t + 1)
        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        alpha_map = np.zeros(shape=(H,W))
        for i,bbox in enumerate(bboxes):
            alpha_map_this_bbox = np.zeros(shape=(H,W))
            xmin,ymin,xmax,ymax = bbox
            xmin_ = int(np.floor(xmin))
            ymin_ = int(np.floor(ymin))
            xmax_ = int(np.ceil(xmax))
            ymax_ = int(np.ceil(ymax))
            alpha_map_this_bbox[ymin_:ymax_,xmin_:xmax_] = current_alpha[i]
            alpha_map += alpha_map_this_bbox

        alpha = skimage.transform.resize(alpha_map, [img_size, img_size])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()