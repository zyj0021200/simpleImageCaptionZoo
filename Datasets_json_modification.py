import os
import argparse
import nltk
import json
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

PUNCTUATIONS = ["''", "'", "``", "`", "[", "]", "(", ")", "{", "}", \
                ".", "?", "!", ",", ":", "-", "--", "...", ";", ">"]

def show_original_COCO14_annotations_example():
    cocofileori = open('./Datasets/MSCOCO/2014/annotations/captions_train2014.json', 'r')
    cocojsori = json.load(cocofileori)
    print(cocojsori.keys())
    img = cocojsori['images']
    print(img[0])
    ann = cocojsori['annotations']
    print(ann[0])

def show_modified_jsons():
    file = open('./Datasets/MSCOCO/2017/modified_annotations/captions_train.json','r')
    js = json.load(file)
    print(js.keys())
    print(js['images'][0])
    print(js['annotations'][0])

    anns = js['annotations']
    for ann in anns:
        if '#' in ann['caption']:
            print(ann['caption'])
            print(ann['tokens'])
            print(ann['file_name'])

def rawCaption2Tokens(raw_cap,tokenizer='nltk',karpathy_data=None,PTB_data=None):
    tokens = None
    if tokenizer == 'karpathy' and karpathy_data is not None:
        tokens = karpathy_data
    elif tokenizer == 'nltk':
        tokens = nltk.word_tokenize(raw_cap.lower())
    elif tokenizer == 'nltk_punc':
        raw_tokens = nltk.word_tokenize(raw_cap.lower())
        tokens = [token for token in raw_tokens if token not in PUNCTUATIONS]
    elif tokenizer == 'PTB' and PTB_data is not None:
        tokens = PTB_data.split(' ')
    return tokens

def generate_modified_json_coco14(ks_json_path,dataset_name='COCO14',tokenizer='karpathy'):
    cocofile = open(ks_json_path, 'r')
    cocojs = json.load(cocofile)
    print('Original images json:')
    print(cocojs['images'][0])
    PTB_rlt = None
    if tokenizer == 'PTB':
        PTB_rlt = PTB_pre_tokenization(json_path=ks_json_path)

    split = ['train', 'val', 'test']
    new_json = {}
    for subset in split:
        new_json[subset] = {'dataset': dataset_name, 'images': [], 'annotations': []}

    for i, img in enumerate(cocojs['images']):
        if img['split'] in ['train', 'restval']:
            split_this_img = 'train'
        else:
            split_this_img = img['split']
        new_img_entry = {}
        new_img_entry.update({'file_name': img['filename']})
        new_img_entry.update({'id': img['cocoid']})
        new_img_entry.update({'sentids': img['sentids']})
        new_sentences = []
        for sid in range(len(img['sentids'])):
            new_ann_entry = {}
            new_ann_entry.update({'file_name': img['filename']})
            new_ann_entry.update({'image_id': img['cocoid']})
            new_ann_entry.update({'id': img['sentids'][sid]})
            new_ann_entry.update({'caption': img['sentences'][sid]['raw']})
            if PTB_rlt is not None:PTB_rlt_this_sent = PTB_rlt[img['sentids'][sid]][0]
            else:PTB_rlt_this_sent = None
            new_filtered_tokens = rawCaption2Tokens(raw_cap=img['sentences'][sid]['raw'],
                                                    tokenizer=tokenizer,
                                                    karpathy_data=img['sentences'][sid]['tokens'],
                                                    PTB_data=PTB_rlt_this_sent)
            new_ann_entry.update({'tokens': new_filtered_tokens})
            new_json[split_this_img]['annotations'].append(new_ann_entry)
            tmp = {'tokens': new_filtered_tokens, 'raw': img['sentences'][sid]['raw']}
            new_sentences.append(tmp)
        new_img_entry.update({'sentences': new_sentences})
        new_json[split_this_img]['images'].append(new_img_entry)

    split = ['train', 'val', 'test']
    output_dir = ks_json_path[:ks_json_path.rfind('/')]     #'./Datasets/MSCOCO/2014'
    os.makedirs(os.path.join(output_dir,'modified_annotations'),exist_ok=True)
    for subset in split:
        json.dump(new_json[subset], open(os.path.join(output_dir,'modified_annotations/captions_%s.json' % subset), 'w'))
    print('Finish generating modified json for Dataset:%s' % dataset_name)
    print('Testing new json..')
    print(new_json['train']['images'][0])
    print(new_json['train']['annotations'][0])

def generate_modified_json_flickr(ks_json_path,dataset_name,tokenizer):
    flickrfile = open(ks_json_path, 'r')
    flickrjs = json.load(flickrfile)
    print('Original images json:')
    print(flickrjs['images'][0])
    PTB_rlt = None
    if tokenizer == 'PTB':
        PTB_rlt = PTB_pre_tokenization(json_path=ks_json_path)

    split = ['train', 'val', 'test']
    new_json = {}
    for subset in split:
        new_json[subset] = {'dataset': dataset_name, 'images': [], 'annotations': []}

    for i, img in enumerate(flickrjs['images']):
        split_this_img = img['split']
        new_img_entry = {}
        new_img_entry.update({'file_name': img['filename']})
        new_img_entry.update({'id': img['imgid']})
        new_img_entry.update({'sentids': img['sentids']})
        new_sentences = []
        for sid in range(len(img['sentids'])):
            new_ann_entry = {}
            new_ann_entry.update({'file_name': img['filename']})
            new_ann_entry.update({'image_id': img['imgid']})
            new_ann_entry.update({'id': img['sentids'][sid]})
            new_ann_entry.update({'caption': img['sentences'][sid]['raw']})
            if PTB_rlt is not None:PTB_rlt_this_sent = PTB_rlt[img['sentids'][sid]][0]
            else:PTB_rlt_this_sent = None
            new_filtered_tokens = rawCaption2Tokens(raw_cap=img['sentences'][sid]['raw'],
                                                    tokenizer=tokenizer,
                                                    karpathy_data=img['sentences'][sid]['tokens'],
                                                    PTB_data=PTB_rlt_this_sent)
            new_ann_entry.update({'tokens': new_filtered_tokens})
            new_json[split_this_img]['annotations'].append(new_ann_entry)
            tmp = {'tokens': new_filtered_tokens, 'raw': img['sentences'][sid]['raw']}
            new_sentences.append(tmp)
        new_img_entry.update({'sentences': new_sentences})
        new_json[split_this_img]['images'].append(new_img_entry)

    split = ['train', 'val', 'test']
    output_dir = ks_json_path[:ks_json_path.rfind('/')]     #'./Datasets/Flickr/30K'
    os.makedirs(os.path.join(output_dir,'modified_annotations'),exist_ok=True)
    for subset in split:
        json.dump(new_json[subset], open(os.path.join(output_dir,'modified_annotations/captions_%s.json' % subset), 'w'))
    print('Finish generating modified json for Dataset:%s' % dataset_name)
    print('Testing new json..')
    print(new_json['train']['images'][0])
    print(new_json['train']['annotations'][0])

def generate_modified_json_coco17(original_json_root,split='train',dataset_name='COCO17',tokenizer='nltk'):
    cocofileori = open(os.path.join(original_json_root,'captions_%s2017.json' % split), 'r')
    cocojsori = json.load(cocofileori)
    print(cocojsori['images'][0])
    print(cocojsori['annotations'][0])
    split = ['train', 'val']
    new_json = {}
    output_dir = './Datasets/MSCOCO/2017/'
    os.makedirs(os.path.join(output_dir,'modified_annotations'),exist_ok=True)
    for subset in split:
        cocofileori = open(os.path.join(original_json_root, 'captions_%s2017.json' % subset), 'r')
        PTB_rlt = None
        if tokenizer == 'PTB':
            PTB_rlt = PTB_pre_tokenization_COCO17(json_path=os.path.join(original_json_root, 'captions_%s2017.json' % subset))
        cocojsori = json.load(cocofileori)
        new_json[subset] = {'dataset': dataset_name, 'images': [], 'annotations': []}
        ori_anns = cocojsori['annotations']
        ori_imgs = cocojsori['images']
        ori_img_dict = {}
        new_img_dict = {}
        for ori_img in ori_imgs:
            new_img_entry = {}
            new_img_entry.update({'file_name':ori_img['file_name']})
            new_img_entry.update({'id':ori_img['id']})
            new_img_entry.update({'sentids':[]})
            new_img_entry.update({'sentences':[]})
            new_img_dict.update({ori_img['id']:new_img_entry})
            ori_img_dict.update({ori_img['id']:ori_img})
        for ori_ann in ori_anns:
            new_ann_entry = {}
            img_id = ori_ann['image_id']
            related_ori_img_entry = ori_img_dict[img_id]
            related_new_img_entry = new_img_dict[img_id]
            related_new_img_entry['sentids'].append(ori_ann['id'])
            new_ann_entry.update({'file_name':related_ori_img_entry['file_name']})
            new_ann_entry.update({'image_id':ori_ann['image_id']})
            new_ann_entry.update({'id':ori_ann['id']})
            new_ann_entry.update({'caption':ori_ann['caption']})
            if PTB_rlt is not None:PTB_rlt_this_sent = PTB_rlt[ori_ann['id']][0]
            else:PTB_rlt_this_sent = None
            new_filtered_tokens = rawCaption2Tokens(raw_cap=ori_ann['caption'],tokenizer=tokenizer,PTB_data=PTB_rlt_this_sent)
            new_ann_entry.update({'tokens':new_filtered_tokens})
            new_json[subset]['annotations'].append(new_ann_entry)
            related_new_img_entry['sentences'].append({'tokens':new_filtered_tokens,'raw':ori_ann['caption']})
        for new_img in list(new_img_dict.values()):
            new_json[subset]['images'].append(new_img)

        json.dump(new_json[subset], open(os.path.join(output_dir,'modified_annotations/captions_%s.json' % subset), 'w'))

    print('Finish generating modified json for Dataset:%s' % dataset_name)
    print('Testing new json..')
    print(new_json['train']['images'][0])
    print(new_json['train']['annotations'][0])

def PTB_pre_tokenization(json_path):
    file = open(json_path,'r')
    js = json.load(file)
    source = {}
    for i,img in enumerate(js['images']):
        for sid in range(len(img['sentids'])):
            tmp = {img['sentids'][sid]:[{'caption':img['sentences'][sid]['raw']}]}
            source.update(tmp)
    tool = PTBTokenizer()
    out = tool.tokenize(captions_for_image=source)
    return out

def PTB_pre_tokenization_COCO17(json_path):
    file = open(json_path,'r')
    js = json.load(file)
    source = {}
    for i,ann in enumerate(js['annotations']):
        tmp = {ann['id']:[{'caption':ann['caption']}]}
        source.update(tmp)
    tool = PTBTokenizer()
    out = tool.tokenize(captions_for_image=source)
    return out

def main(args):
    #generate_modified_json_flickr(ks_json_path=args.Flickr30K_karpathy_json_path,dataset_name='Flickr30K',tokenizer=args.tokenizer)
    #generate_modified_json_flickr(ks_json_path=args.Flickr8K_karpathy_json_path,dataset_name='Flickr8K',tokenizer=args.tokenizer)
    #generate_modified_json_coco14(ks_json_path=args.COCO14_karpathy_json_path,dataset_name='COCO14',tokenizer=args.tokenizer)
    #generate_modified_json_coco17(original_json_root=args.COCO17_original_json_root,tokenizer=args.tokenizer)
    show_modified_jsons()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------global settings-------------------#
    parser.add_argument('--dataset',type=str,default='Flickr30K')
    parser.add_argument('--Flickr8K_karpathy_json_path',type=str,default='./Datasets/Flickr/8K/dataset_flickr8k.json')
    parser.add_argument('--Flickr30K_karpathy_json_path',type=str,default='./Datasets/Flickr/30K/dataset_flickr30k.json')
    parser.add_argument('--COCO14_karpathy_json_path',type=str,default='./Datasets/MSCOCO/2014/dataset_coco.json')
    parser.add_argument('--COCO17_original_json_root', type=str, default='./Datasets/MSCOCO/2017/annotations/')
    parser.add_argument('--tokenizer',type=str,default='PTB')
    args = parser.parse_args()
    main(args)