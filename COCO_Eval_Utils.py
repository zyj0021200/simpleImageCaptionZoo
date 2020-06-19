import json
import os
import pickle
import torch
import torch.utils.data as tdata
from torchvision import transforms
import skimage.io as io
from PIL import Image
from pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap
import nltk
import matplotlib.pyplot as plt
import numpy as np

def coco_eval(results,eval_caption_path):
    eval_json_output_dir = './coco_caption/results/'
    os.makedirs(eval_json_output_dir,exist_ok=True)
    resFile = eval_json_output_dir + 'captions-generate.json'
    json.dump(results,open(resFile,'w'))

    annFile = eval_caption_path
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    cocoEval = COCOEvalCap(coco,cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    cider = 0
    print('---------------Evaluation performance-----------------')
    for metric,score in cocoEval.eval.items():
        print('%s: %.3f'%(metric,score))
        if metric == 'CIDEr':
            cider = score
    return cider

def coco_eval_specific(results,eval_caption_path,entry_limit=500):
    eval_json_output_dir = './coco/results/'
    os.makedirs(eval_json_output_dir,exist_ok=True)
    resFile = eval_json_output_dir + 'captions-generate.json'
    json.dump(results,open(resFile,'w'))

    annFile = eval_caption_path
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)

    cocoEval = COCOEvalCap(coco,cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    ans = [{'img_id':eva['image_id'],'CIDEr':eva['CIDEr']} for eva in cocoEval.evalImgs]
    os.makedirs('./Data/Eval_Statics/',exist_ok=True)
    with open("./Data/Eval_Statics/CIDEr_Result.txt",'w') as f:
        entry = "img_id" + " " + "CIDEr" + "\n"
        f.writelines(entry)
        entry_num = 0
        for ans_entry in ans:
            entry = str(ans_entry['img_id']) + " " + str(np.round(ans_entry['CIDEr'],2)) + "\n"
            f.writelines(entry)
            entry_num += 1
            if entry_num >= entry_limit: break
        cider_list = [eva['CIDEr'] for eva in cocoEval.evalImgs]
        cider_list_npy = np.array(cider_list)
        indices = np.argsort(cider_list_npy)[::-1]
        f.writelines('best samples:\n')
        for idx in indices[:50]:
            entry = str(ans[idx]['img_id']) + " " + str(np.round(ans[idx]['CIDEr'],2)) + "\n"
            f.writelines(entry)
        indices = indices[::-1]
        f.writelines('worst samples:\n')
        for idx in indices[:50]:
            entry = str(ans[idx]['img_id']) + " " + str(np.round(ans[idx]['CIDEr'],2)) + "\n"
            f.writelines(entry)

    f.close()

    ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]

    x = plt.hist(ciderScores,bins=[0,1,2,3,4,5,6,7,8,9,10])
    print(x)
    plt.title('Histogram of CIDEr Scores', fontsize=20)
    plt.xlabel('CIDEr score', fontsize=20)
    plt.ylabel('result counts', fontsize=20)
    plt.savefig('ciderHist.png',dpi=500)
    plt.show()
