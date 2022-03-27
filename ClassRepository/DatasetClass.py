import time
import json
from collections import defaultdict

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class CaptionData:
    def __init__(self,annotation_file=None):
        self.dataset,self.imgs,self.anns = dict(),dict(),dict()
        if not annotation_file is None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, imgs = {}, {}
        imgToAnns = defaultdict(list)
        filenameToImgid = {}                           #for locating file
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)          #for using cocoeval
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
                filenameToImgid.update({img['file_name']:img['id']})

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.filenameToImgid = filenameToImgid

