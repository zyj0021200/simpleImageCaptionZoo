import h5py
import base64
import sys
import argparse
import csv
import os
from tqdm import tqdm
import pickle
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
TOTAL_TRAINVAL_NUM = 123287
NUM_FIXED_BBOXS = 36
FEATURE_DIM = 2048

def generate(args):
    trainval_data_file = os.path.join(args.COCO14_output_dir,'trainval_36.hdf5')
    h_file = h5py.File(trainval_data_file,'w')
    img_features = h_file.create_dataset(
        name='image_features',shape=(TOTAL_TRAINVAL_NUM,NUM_FIXED_BBOXS,FEATURE_DIM),dtype='f'
    )
    img_bboxs = h_file.create_dataset(
        name='image_bboxes',shape=(TOTAL_TRAINVAL_NUM,NUM_FIXED_BBOXS,4),dtype='f'
    )
    img_indices = {}
    with open(args.COCO14_trainval_36_path,'r') as tsv_file:
        reader = csv.DictReader(tsv_file,delimiter='\t',fieldnames=FIELDNAMES)
        cnt = 0
        for item in tqdm(reader):
            num_boxes = int(item['num_boxes'])
            img_id = int(item['image_id'])
            boxes = np.frombuffer(
                base64.b64decode(item['boxes']),dtype=np.float32
            ).reshape(num_boxes,-1)         #(36,4(xmin,ymin,xmax,ymax))
            features = np.frombuffer(
                base64.b64decode(item['features']),dtype=np.float32
            ).reshape(num_boxes,-1)         #(36,2048)
            img_indices[img_id] = cnt
            img_features[cnt,:,:] = features
            img_bboxs[cnt,:,:] = boxes
            cnt += 1

    h_file.close()
    pickle.dump(img_indices,open(os.path.join(args.COCO14_output_dir,'trainval_36_indices.pkl'),'wb'))

def check(args):
    trainval_data_file = os.path.join(args.COCO14_output_dir,'trainval_36.hdf5')
    h_file = h5py.File(trainval_data_file,'r')
    indices = pickle.load(open(os.path.join(args.COCO14_output_dir,'trainval_36_indices.pkl'),'rb'))
    train_json = json.load(open(args.COCO14_train_json,'r'))
    img_entry = train_json['images'][316]
    ann_entry = train_json['annotations'][17]
    print('example image json')
    print(img_entry)
    print('example ann json')
    print(ann_entry)
    img_filename = img_entry['file_name']
    img_id = img_entry['id']

    if indices.__contains__(184613):
        print(indices[184613])
    if 'train' in img_filename.lower():
        img_path = os.path.join(args.COCO14_img_root, 'train2014', img_filename)
    else:
        img_path = os.path.join(args.COCO14_img_root, 'val2014', img_filename)
    original_img = Image.open(img_path).convert('RGB')
    img_ind = indices[img_id]
    bbox = h_file['image_bboxes'][img_ind]

    plt.imshow(original_img)
    plt.title('Bottom-up-features bboxes for COCO14 Image:\n%s' % (img_filename), fontsize=20)
    plt.axis('off')
    if len(bbox) > 0:
        for i, bb in enumerate(bbox[:10]):
            xmin, ymin, xmax, ymax = bb
            plt.gca().add_patch(
                plt.Rectangle(xy=(xmin, ymin), height=(ymax-ymin), width=(xmax-xmin), fill=False, edgecolor='r', linewidth=2)
            )
    plt.show()

def main(args):
    if args.operation == 'generate':
        generate(args=args)
    if args.operation == 'check':
        check(args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------------------------------Global Settings------------------------------------#
    parser.add_argument('--COCO14_trainval_36_path',type=str,default='./Datasets/MSCOCO/2014/bottom_up_features/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv')
    parser.add_argument('--COCO14_train_json',type=str,default='./Datasets/MSCOCO/2014/modified_annotations/captions_train.json')
    parser.add_argument('--COCO14_val_json',type=str,default='./Datasets/MSCOCO/2014/modified_annotations/captions_val.json')
    parser.add_argument('--COCO14_img_root',type=str,default='./Datasets/MSCOCO/2014/')
    parser.add_argument('--COCO14_output_dir',type=str,default='./Data/MSCOCO/2014/')
    parser.add_argument('--operation',type=str,default='check')
    args = parser.parse_args()
    main(args)