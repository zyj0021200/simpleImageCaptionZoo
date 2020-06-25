import os
import argparse
import torch
from ModelEngines.NIC_Engine import NIC_Eng
from ModelEngines.BUTD_Engine import BUTDSpatial_Eng,BUTDDetection_Eng
from ModelEngines.AoA_Engine import AoASpatial_Eng,AoADetection_Eng
from Utils import parse_data_config,get_caption_vocab,get_train_dataloader,get_eval_dataloader,get_scst_train_dataloader

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(4096,rlimit[1]))

def main(args):
    #--------------------------- loading_cfg-------------------------------------#
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # Add the absolute path of the project to avoid possible path indexing problems later
    config_path = os.path.join(args.dataset_config_root,args.dataset+'.data')
    # opt contains the path/dir/root info. of different datasets
    opt = parse_data_config(config_path,base_dir)
    #----------------------------device settings---------------------------------#
    # Currently only supports single GPU
    device = args.gpu_id if torch.cuda.is_available() else 'cpu'
    #------------------------Caption Vocab Preprocess----------------------------#
    # Mkdir for storing supplementary data for different datasets, e.g. the caption_vocab, bottom_up_features will be dumped here.
    os.makedirs(opt['data_dir'],exist_ok=True)
    # Build/Load caption vocab for different datasets
    caption_vocab = get_caption_vocab(args,opt)
    # ------------------------ModelEngine_init----------------------------------#
    # Build up models based on the MODELNAME.json stored in Configs/Models/
    # 'Detecton' means the models need the pretrained bottom-up features,
    # currently only support COCO14 bottom-up-features,
    # thus only valid for training and testing of COCO14 Dataset
    if args.model_type == 'NIC':
        model = NIC_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                        dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    elif args.model_type == 'BUTDSpatial':
        model = BUTDSpatial_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                                dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    elif args.model_type == 'BUTDDetection':
        model = BUTDDetection_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                                  dataset_name=args.dataset,caption_vocab=caption_vocab,data_dir=opt['data_dir'],device=device)
    elif args.model_type == 'AoASpatial':
        model = AoASpatial_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                               dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    elif args.model_type == 'AoADetection':
        model = AoADetection_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                                 dataset_name=args.dataset, caption_vocab=caption_vocab, data_dir=opt['data_dir'], device=device)
    print('model construction complete.')
    #---------------------------Operations---------------------------------------#
    if args.operation == 'train':
        train_dataloader = get_train_dataloader(args=args,opt=opt,caption_vocab=caption_vocab)
        # Note that Flickr8K/30K/COCO14KS contains 'val' and 'test' split, we only evaluate on the 'val' split after each training epoch
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val')
        # Please refer to the argparse_part for the function description of some parameters
        model.training(
            num_epochs=args.num_epochs,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],      # evaluation needs the raw captions stored in the json_file
            optimizer_type=args.optimizer,
            lr_opts={'learning_rate':args.learning_rate,                    # learning rate & lr decay settings
                     'cnn_FT_learning_rate':args.cnn_FT_learning_rate,      # only valid for models using cnn extractors like ResNet-101(NIC,BUTDSpatial,AoASpatial)
                     'lr_dec_start_epoch':args.learning_rate_decay_start,
                     'lr_dec_every':args.learning_rate_decay_every,
                     'lr_dec_rate':args.learning_rate_decay_rate},
            ss_opts={'ss_start_epoch':args.scheduled_sampling_start,        # scheduled sampling settings
                     'ss_inc_every':args.scheduled_sampling_increase_every,
                     'ss_inc_prob':args.scheduled_sampling_increase_prob,
                     'ss_max_prob':args.scheduled_sampling_max_prob},
            use_preset_settings=args.use_preset,        # decide whether to use the preset lr&optimizer settings stored in MODELNAME.json
            eval_beam_size=args.eval_beam_size,         # whether to eval with beam search strategy
            load_pretrained_model=args.load_pretrained_model,   # you could restart training from the checkpoint
            overwrite_guarantee=args.overwrite_guarantee,       # prevent previously-stored best checkpoint from being overwritten
            cnn_FT_start=args.cnn_FT_start,             # start CNN_Fine_tune immediately, mainly for restarting from suspension
            tqdm_visible=args.tqdm_visible              # enable the tqdm_bar to show the training/evaluation process
        )
    if args.operation == 'scst_train':
        # The operation is basically the same as the above, except to change the train_dataloader for SCST
        scst_train_dataloader = get_scst_train_dataloader(args=args,opt=opt)
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val')
        # Parameter settings similar to the model training part
        # Note that we default the cnn_FT_start=True when start scst training since the models have been pretrained
        model.SCSTtraining(
            num_epochs=args.scst_num_epochs,
            train_dataloader=scst_train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],
            optimizer_type=args.optimizer,
            scst_lr=args.scst_learning_rate,
            scst_cnn_FT_lr=args.scst_cnn_FT_learning_rate,
            use_preset_settings=args.use_preset,
            eval_beam_size=args.eval_beam_size,
            load_pretrained_scst_model=args.load_pretrained_scst_model,
            overwrite_guarantee=args.overwrite_guarantee,
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'eval':
        # As mentioned before, you can choose to evaluate on the 'val' or 'test' split for Flickr/COCO14KS dataset
        # Note that we do not have the annotations for COCO17 test split, thus evaluation of COCO17 test split is not available.
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split=args.eval_split)
        if args.eval_split == 'val':eval_caption_path = opt['val_caption_path']
        else:eval_caption_path = opt['test_caption_path']
        model.eval(
            dataset=args.dataset,
            split=args.eval_split,
            eval_scst=args.eval_scst,           # decide whether to evaluate models trained under scst optimization
            eval_dataloader=eval_dataloader,
            eval_caption_path=eval_caption_path,
            eval_beam_size=args.eval_beam_size, # decide whether to use beam search strategy when evaluating
            output_statics=False,               # when this option is set True, specific cider score results will be output
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'sample':
        model.test(
            use_scst_model=args.eval_scst,
            img_root=args.COCO14_img_root,
            img_filename='COCO_val2014_000000356708.jpg',
            eval_beam_size=args.eval_beam_size
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------------------------------Global Settings------------------------------------#
    parser.add_argument('--dataset',type=str,default='COCO14',help='choose the dataset for training and evaluating')
    parser.add_argument('--model_type',type=str,default='BUTDDetection',help='choose the model_type, currently only supports two models')
    parser.add_argument('--dataset_config_root',type=str,default='./Configs/Datasets/',help='root to store dataset_configs')
    parser.add_argument('--model_config_root',type=str,default='./Configs/Models/',help='root to store model_configs')
    parser.add_argument('--gpu_id',type=str,default='cuda:6')
    parser.add_argument('--tqdm_visible',type=bool,default=True,help='choose to enable the tqdm_bar to show the training/evaluation process')
    parser.add_argument('--operation',type=str,default='train')

    #-----------------------------------Train Settings---------------------------------------#
    # Note that models trained w/w_o SCST algorithm are stored separately
    #------------------------------global training settings----------------------------------#
    parser.add_argument('--cnn_FT_start',type=bool,default=False,help='enable CNN fine-tune immediately, this option is mainly used for restarting after training suspension')
    parser.add_argument('--overwrite_guarantee',type=bool,default=True,help='when this option is set True, the subsequent training process will only save the model checkpoints better than the previously stored records')
    parser.add_argument('--img_size',type=int,default=224)
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--use_preset',type=bool,default=True,help='when this option is set True, the model will be trained under the preset lr and optimizer in "MODELNAME.json"')
    #-----------------------------XE Loss training settings----------------------------------#
    parser.add_argument('--load_pretrained_model',type=bool,default=False,help='decide whether to load the pretrained checkpoints when starting a new training process')
    parser.add_argument('--num_epochs',type=int,default=50,help='maximum training epochs for training under XE Loss')
    parser.add_argument('--train_batch_size',type=int,default=128)
    parser.add_argument('--learning_rate',type=float,default=5e-4)
    parser.add_argument('--cnn_FT_learning_rate',type=float,default=5e-5,help='only valid for models using CNN to extract image features')
    parser.add_argument('--scheduled_sampling_start',type=int,default=0,help='when this option is set -1, scheduled sampling is disabled')
    parser.add_argument('--scheduled_sampling_increase_every',type=int,default=5)
    parser.add_argument('--scheduled_sampling_increase_prob',type=float,default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob',type=float,default=0.25)
    parser.add_argument('--learning_rate_decay_start',type=int,default=0,help='when this option is set -1,lr decay is disabled')
    parser.add_argument('--learning_rate_decay_every',type=int,default=5)
    parser.add_argument('--learning_rate_decay_rate',type=float,default=0.8)
    #--------------------------------SCST Training settings------------------------------------#
    parser.add_argument('--load_pretrained_scst_model',type=bool,default=True,help='decide whether to load the pretrained scst chekcpoints when starting a new scst training process')
    parser.add_argument('--scst_num_epochs',type=int,default=30,help='maximum training epochs for training under SCST strategy')
    parser.add_argument('--scst_train_batch_size',type=int,default=64)
    parser.add_argument('--scst_learning_rate',type=float,default=2e-5)
    parser.add_argument('--scst_cnn_FT_learning_rate',type=float,default=1e-5)

    #---------------------------------Evaluating Settings---------------------------------------#
    parser.add_argument('--eval_scst',type=bool,default=False,help='choose whether evaluating/testing on the model trained under SCST algorithm')
    parser.add_argument('--eval_split',type=str,default='test',help="since Flickr8K/30K/COCO14(KarpathySplitVer) contains test split, you can choose to eval on test split")
    parser.add_argument('--eval_batch_size',type=int,default=64)
    parser.add_argument('--eval_beam_size',type=int,default=-1,help='when this option is set positive(e.g. =3), the beam search sampling is enabled during evaluating or testing')

    #------------------------------Testing(sampling) Settings------------------------------------#
    # these img_root settings are mainly for testing images from the original datasets
    # You can also test your own images by putting the images under Sample_img_root
    parser.add_argument('--Flickr8K_img_root',type=str,default='./Datasets/Flickr/8K/images/')
    parser.add_argument('--Flickr30K_img_root',type=str,default='./Datasets/Flickr/30K/images/')
    parser.add_argument('--COCO14_img_root',type=str,default='./Datasets/MSCOCO/2014/')
    parser.add_argument('--COCO17_img_root', type=str, default='./Datasets/MSCOCO/2017/')
    parser.add_argument('--Sample_img_root',type=str,default='./Data/Sample_images/')

    args = parser.parse_args()
    main(args)