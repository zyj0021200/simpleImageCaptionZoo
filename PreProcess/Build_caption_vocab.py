import argparse
from collections import Counter
import pickle
import os
from ClassRepository.DatasetClass import CaptionData
from ClassRepository.CaptionVocabClass import Caption_Vocabulary

file_path = {
    'Flickr8K_train_json_path':'../Datasets/Flickr/8K/modified_annotations/captions_train.json',
    'Flickr30K_train_json_path':'../Datasets/Flickr/30K/modified_annotations/captions_train.json',
    'COCO14_train_json_path':'../Datasets/MSCOCO/2014/modified_annotations/captions_train.json',
    'COCO17_train_json_path':'../Datasets/MSCOCO/2017/modified_annotations/captions_train.json'
}

output_dir = {
    'Flickr8K_output_dir':'../Data/Flickr/8K/',
    'Flickr30K_output_dir':'../Data/Flickr/30K/',
    'COCO14_output_dir':'../Data/MSCOCO/2014/',
    'COCO17_output_dir':'../Data/MSCOCO/2017/'
}

def build_vocab(json_path, threshold):
    capdata = CaptionData(json_path)
    ids = capdata.anns.keys()
    counter = Counter()

    for i, id in enumerate(ids):
        tokens = capdata.anns[id]['tokens']
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." % (i, len(ids)))

    # we only add the word occurred more than 'threshold' times to the vocabulary
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    vocab = Caption_Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<sta>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

def main(args):
    train_caption_path = file_path['%s_train_json_path' % args.dataset]
    vocab_output_dir = output_dir['%s_output_dir' % args.dataset]
    os.makedirs(vocab_output_dir,exist_ok=True)
    caption_vocab_path = os.path.join(vocab_output_dir,'caption_vocab.pkl')
    caption_vocab = build_vocab(json_path=train_caption_path,threshold=args.threshold)
    with open(caption_vocab_path, 'wb') as f:
        pickle.dump(caption_vocab, f)
    print("Total vocabulary size: %d" % len(caption_vocab))
    print("Saved the vocabulary wrapper to '%s'" % caption_vocab_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='Flickr30K')
    parser.add_argument('--threshold',type=int,default=5)
    args = parser.parse_args()
    main(args)
