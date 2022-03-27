import base64
import sys
import argparse
import csv
import os
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

csv.field_size_limit(2000000)   #'sys.maxsize' also works in Linux environment

# adaptive infiles contain 10 to 100 features per image
infiles_adaptive = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv',
                    'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
                    'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
                    'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']
# fixed infiles contain 36 features per image
infiles_fixed = ['trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv']

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

def generate(tsv_root,bu_type,output_dir):
    '''
    Extract the bottom_up features and bboxes info. for each image from the tsv file, and save the data in numpy-format.
    :param tsv_root: data_root of the tsv files
    :param bu_type: choose 'fixed' or 'adaptive'
    :param output_dir: dir to output the npz/npy files
    :return: None
    '''
    if bu_type == 'fixed':
        infiles = infiles_fixed
    else:
        infiles = infiles_adaptive

    # bu_feat contains the (num_bbox,2048) image features
    # bu_bbox contains the (num_bbox,4) bboxes information for visualization or other use
    bu_feat_output_dir = os.path.join(output_dir,'%s_bu_feat' % bu_type)
    bu_bbox_output_dir = os.path.join(output_dir,'%s_bu_bbox' % bu_type)
    os.makedirs(bu_feat_output_dir, exist_ok=True)
    os.makedirs(bu_bbox_output_dir, exist_ok=True)

    for infile in infiles:
        print('Reading ' + infile)
        with open(os.path.join(tsv_root,infile),'r') as tsv_file:
            reader = csv.DictReader(tsv_file,delimiter='\t',fieldnames=FIELDNAMES)
            for item in tqdm(reader):
                num_boxes = int(item['num_boxes'])
                boxes = np.frombuffer(
                    base64.b64decode(item['boxes']),dtype=np.float32
                ).reshape(num_boxes,-1)         #(36/10~100,4(xmin,ymin,xmax,ymax))
                features = np.frombuffer(
                    base64.b64decode(item['features']),dtype=np.float32
                ).reshape(num_boxes,-1)         #(36/10~100,2048)
                np.savez_compressed(os.path.join(bu_feat_output_dir, str(item['image_id'])),
                                    feat=features)
                np.save(os.path.join(bu_bbox_output_dir, str(item['image_id'])), boxes)
        print('Finish saving bu_features from ' + infile)

def check(check_json_path,data_dir,bu_type,img_root):
    '''
    Visualize the bboxes in the bottom_up features corresponding to an image
    :param check_json_path: json file to load
    :param data_dir: dir where the bu_features is stored
    :param bu_type: choose 'fixed' or 'adaptive'
    :param img_root: COCO14 image root
    :return:
    '''
    # Again we can output the modified json entries to see the data json format
    # I randomly choose the index of an image_entry in the json file.
    train_json = json.load(open(check_json_path,'r'))
    img_entry = train_json['images'][316]
    print('example image json')
    print(img_entry)
    img_filename = img_entry['file_name']
    img_id = img_entry['id']

    bu_feat_output_dir = os.path.join(data_dir,'%s_bu_feat' % bu_type)
    bu_bbox_output_dir = os.path.join(data_dir,'%s_bu_bbox' % bu_type)

    if 'train' in img_filename.lower():
        img_path = os.path.join(img_root, 'train2014', img_filename)
    else:
        img_path = os.path.join(img_root, 'val2014', img_filename)
    original_img = Image.open(img_path).convert('RGB')

    bu_feat = np.load(os.path.join(bu_feat_output_dir,'%s.npz' % (str(img_id))))['feat']
    print('bottom_up_features shape:')
    print(bu_feat.shape)
    bu_bbox = np.load(os.path.join(bu_bbox_output_dir,'%s.npy' % (str(img_id))))

    plt.imshow(original_img)
    plt.title('Bottom-up-features(%s) bboxes for COCO14 Image:\n%s' % (bu_type,img_filename), fontsize=14)
    plt.axis('off')
    if len(bu_bbox) > 0:
        for i, bb in enumerate(bu_bbox):
            xmin, ymin, xmax, ymax = bb
            plt.gca().add_patch(
                plt.Rectangle(xy=(xmin, ymin), height=(ymax-ymin), width=(xmax-xmin), fill=False, edgecolor='r', linewidth=2)
            )
    plt.savefig('%s_bu_bbox_visualization.png' % bu_type)
    plt.show()

def main(args):
    if args.operation == 'generate':
        if args.bu_type == 'fixed':
            tsv_root = args.COCO14_trainval_fixed_root
        else:
            tsv_root = args.COCO14_trainval_adaptive_root
        generate(tsv_root=tsv_root,bu_type=args.bu_type,output_dir=args.COCO14_output_dir)
    if args.operation == 'check':
        check(check_json_path=args.COCO14_train_json,data_dir=args.COCO14_output_dir,bu_type=args.bu_type,img_root=args.COCO14_img_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------------------------------Global Settings------------------------------------#
    parser.add_argument('--COCO14_trainval_fixed_root',type=str,default='../Datasets/MSCOCO/2014/bottom_up_features/fixed/')
    parser.add_argument('--COCO14_trainval_adaptive_root',type=str,default='../Datasets/MSCOCO/2014/bottom_up_features/adaptive/')
    parser.add_argument('--COCO14_train_json',type=str,default='../Datasets/MSCOCO/2014/modified_annotations/captions_train.json')
    parser.add_argument('--COCO14_val_json',type=str,default='../Datasets/MSCOCO/2014/modified_annotations/captions_val.json')
    parser.add_argument('--COCO14_img_root',type=str,default='../Datasets/MSCOCO/2014/')
    parser.add_argument('--COCO14_output_dir',type=str,default='../Data/MSCOCO/2014/')
    parser.add_argument('--bu_type',type=str,default='fixed',help='choose to use the fixed bu_features (36 roi_feats for each image) or the adaptive bu_features (10 to 100 roi_feats for each image)')
    parser.add_argument('--operation',type=str,default='check',help='choose from "generate" and "check"')
    args = parser.parse_args()
    main(args)