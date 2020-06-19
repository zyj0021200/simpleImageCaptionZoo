import os
import argparse
import torch
from Engine import NIC_Eng,BUTDSpatial_Eng
from Utils import parse_data_config,get_caption_vocab,get_train_dataloader,get_eval_dataloader,get_scst_train_dataloader

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(4096,rlimit[1]))

def main(args):
    #-------------------------loading_cfg---------------------------------------#
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Add the absolute path of the project to avoid possible path problems later
    config_path = os.path.join(args.dataset_config_root,args.dataset+'.data')
    # opt contains the path/dir/root info. of different datasets
    opt = parse_data_config(config_path,base_dir)
    #-------------------------device settings-----------------------------------#
    # Currently only supports single GPU
    device = args.gpu_id if torch.cuda.is_available() else 'cpu'
    #-------------------------Caption Vocab Preprocess--------------------------#
    # Mkdir for data storage for different datasets, the caption_vocab will be dumped here
    os.makedirs(opt['data_dir'],exist_ok=True)
    # Build/Load caption vocab for different datasets
    caption_vocab = get_caption_vocab(args,opt)
    # ------------------------ModelEngine_init----------------------------------#
    # Build up Models based on the model_setting_configs
    if args.model_type == 'NIC':
        model = NIC_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                        dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    elif args.model_type == 'BUTDSpatial':
        model = BUTDSpatial_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                        dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    print('model construction complete.')
    #------------------------operations-----------------------------------------#
    if args.operation == 'train':
        train_dataloader = get_train_dataloader(args=args,opt=opt,caption_vocab=caption_vocab)
        # note that Flickr8K/30K/COCO14KS contains 'val' and 'test' split, we only evaluate on the 'val' split after each training epoch
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val')
        # Please refer to the argparse_part for the function description of some parameters
        model.training(
            num_epochs=args.num_epochs,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],
            eval_beam_size=args.eval_beam_size,
            load_pretrained_model=args.load_pretrained_model,
            overwrite_guarantee=args.overwrite_guarantee,
            cnn_FT_start=args.cnn_FT_start,
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'scst_train':
        # The operation is basically the same as the above, except to change the train_dataloader for SCST
        scst_train_dataloader = get_scst_train_dataloader(args=args,opt=opt)
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val')
        model.SCSTtraining(
            num_epochs=args.scst_num_epochs,
            train_dataloader=scst_train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],
            eval_beam_size=args.eval_beam_size,
            load_pretrained_scst_model=args.load_pretrained_scst_model,
            overwrite_guarantee=args.overwrite_guarantee,
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'eval':
        # As mentioned before, you can choose to evaluate on the 'val' or 'test' split for Flickr/COCO14KS dataset
        # note that we do not have the annotations for COCO17 test split, thus evaluation of COCO17 test split is not available.
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split=args.eval_split)
        if args.eval_split == 'val':eval_caption_path = opt['val_caption_path']
        else:eval_caption_path = opt['test_caption_path']
        model.eval(
            dataset=args.dataset,
            split=args.eval_split,
            eval_scst=args.eval_scst,
            eval_dataloader=eval_dataloader,
            eval_caption_path=eval_caption_path,
            eval_beam_size=args.eval_beam_size,
            output_statics=False,
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'sample':
        model.test(
            use_scst_model=args.eval_scst,
            img_root=args.COCO14_img_root,
            img_filename='COCO_val2014_000000314294.jpg',
            eval_beam_size=args.eval_beam_size
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------global settings-------------------#
    parser.add_argument('--dataset',type=str,default='COCO14',help='choose the dataset for training and evaluation')
    #-----------these img_root settings are mainly for testing images from the original datasets--------------#
    #-----------You can also test your own image by putting the image under Sample_img_root-------------------#
    parser.add_argument('--Flickr8K_img_root',type=str,default='./Datasets/Flickr/8K/images/')
    parser.add_argument('--Flickr30K_img_root',type=str,default='./Datasets/Flickr/30K/images/')
    parser.add_argument('--COCO14_img_root',type=str,default='./Datasets/MSCOCO/2014/')
    parser.add_argument('--COCO17_img_root', type=str, default='./Datasets/MSCOCO/2017/')
    parser.add_argument('--Sample_img_root',type=str,default='./Data/Sample_images/')

    parser.add_argument('--model_type',type=str,default='BUTDSpatial',help='choose the model_type, currently only supports two models')
    parser.add_argument('--dataset_config_root',type=str,default='./Configs/Datasets/',help='root to store dataset configs')
    parser.add_argument('--model_config_root',type=str,default='./Configs/Models/',help='root to store model configs')
    parser.add_argument('--gpu_id',type=str,default='cuda:6')
    parser.add_argument('--tqdm_visible',type=bool,default=False,help='choose to enable the tqdm_bar to show the training/evaluation process')
    parser.add_argument('--operation',type=str,default='eval')
    #-----------train settings---------------------#
    #-----------note that models trained w/w_o SCST algorithm are stored separately
    parser.add_argument('--load_pretrained_model',type=bool,default=True,help='decide whether to load the pretrained checkpoints when starting a new training process')
    parser.add_argument('--load_pretrained_scst_model',type=bool,default=True,help='decide whether to load the pretrained scst chekcpoints when starting a new scst training process')
    parser.add_argument('--num_epochs',type=int,default=50,help='maximum training epochs for training under XE Loss')
    parser.add_argument('--scst_num_epochs',type=int,default=30,help='maximum training epochs for training under SCST Loss')
    parser.add_argument('--cnn_FT_start',type=bool,default=False,help='enable CNN Fine tune immediately, this option is mainly used for restarting after training suspension')
    parser.add_argument('--overwrite_guarantee',type=bool,default=True,help='When this option is True, the subsequent training process will only save the model checkpoints better than the previous training results')
    parser.add_argument('--img_size',type=int,default=224)
    parser.add_argument('--train_batch_size',type=int,default=128)
    parser.add_argument('--SCST_train_batch_size',type=int,default=64)

    #-----------eval settings------------------------#
    parser.add_argument('--eval_scst',type=bool,default=True,help='choose whether evaluating on the model trained under SCST algorithm')
    parser.add_argument('--eval_split',type=str,default='test',help="Since Flickr8K/30K/COCO14(KarpathySplitVer) contains test split, you can choose to eval on test split")
    parser.add_argument('--eval_batch_size',type=int,default=64)
    parser.add_argument('--eval_beam_size',type=int,default=5,help='when this option is set positive(e.g. =3), the beam search sampling is enabled during evaluating or testing')

    args = parser.parse_args()
    main(args)