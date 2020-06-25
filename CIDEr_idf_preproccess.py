import argparse
import json
from collections import defaultdict
from six.moves import cPickle
import six

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
        # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency

def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs

def build_dict(imgs):
    refs_words = []
    for img in imgs:
        ref_words = []
        for sent in img['sentences']:
            ref_words.append(' '.join(sent['tokens']))
        refs_words.append(ref_words)
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    return ngram_words

def main(args):
    if args.operation == 'build':
        if args.dataset == 'Flickr30K':
            jsonpath = args.Flickr30K_train_json_path
        elif args.dataset == 'Flickr8K':
            jsonpath = args.Flickr8K_train_json_path
        elif args.dataset == 'COCO14':
            jsonpath = args.COCO14_train_json_path
        elif args.dataset == 'COCO17':
            jsonpath = args.COCO17_train_json_path
        data = json.load(open(jsonpath,'r'))
        imgs = data['images']
        ref_len = len(imgs)
        ngram_words = build_dict(imgs)
        pfile = {'document_frequency':ngram_words,'ref_len':ref_len}
        with open('./cider/data/%s-val.p' % args.dataset ,'wb') as file:
            cPickle.dump(pfile,file,protocol=2)
        print('Finish dumping doc_freq file for dataset:%s.' % args.dataset)
    elif args.operation == 'check':
        pkl_file = cPickle.load(open('./cider/data/COCO14-train.p', 'rb'),
                                **(dict(encoding='latin1') if six.PY3 else {}))
        ref_len = pkl_file['ref_len']
        print('ref_len:%d' % ref_len)
        document_frequency = pkl_file['document_frequency']
        print('doc_freq_len:%d' % len(document_frequency))
        cnt = 0
        for key,value in document_frequency.items():
            #print(key)
            #print(value)
            cnt += 1
            #if cnt > 10: break
            if 'vehicle' in key:
                print(key)
                print(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------global settings-------------------#
    parser.add_argument('--operation',type=str,default='check')
    parser.add_argument('--dataset',type=str,default='COCO14')
    parser.add_argument('--Flickr8K_train_json_path',type=str,default='./Datasets/Flickr/8K/modified_annotations/captions_train.json')
    parser.add_argument('--Flickr30K_train_json_path',type=str,default='./Datasets/Flickr/30K/modified_annotations/captions_train.json')
    parser.add_argument('--COCO14_train_json_path',type=str,default='./Datasets/MSCOCO/2014/modified_annotations/captions_train.json')
    parser.add_argument('--COCO17_train_json_path', type=str, default='./Datasets/MSCOCO/2017/modified_annotations/captions_train.json')
    args = parser.parse_args()
    main(args)