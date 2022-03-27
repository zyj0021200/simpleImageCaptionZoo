import os
import argparse
import torch
import pickle
from ModelEngines.NIC_Engine import NIC_Eng
from ModelEngines.BUTD_Engine import BUTDSpatial_Eng,BUTDDetection_Eng
from ModelEngines.AoA_Engine import AoASpatial_Eng,AoADetection_Eng
from Utils import parse_data_config,get_train_dataloader,get_eval_dataloader,get_scst_train_dataloader

# the following lines should be annotated in Windows environment
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
    device = 'cuda:%s' % args.gpu_id if torch.cuda.is_available() else 'cpu'
    print(device)
    #-------------------------Load Caption Vocab----------------------------#
    # Mkdir for storing supplementary data for different datasets, e.g. the caption_vocab, bottom_up_features will be dumped here.
    os.makedirs(opt['data_dir'],exist_ok=True)
    # Load caption vocab for different datasets
    caption_vocab_path = opt['caption_vocab_path']
    if os.path.exists(caption_vocab_path):
        caption_vocab_file = open(caption_vocab_path, 'rb')
        caption_vocab = pickle.load(caption_vocab_file)
        print('Caption Vocab for dataset:%s loaded complete.' % args.dataset)
    else:
        caption_vocab = None
        print('Caption Vocab not generated. Run PreProcess/Build_caption_vocab.py first.')
    #-------------------------Decide whether using supplementary datas--------------------#
    supp_infos = []
    if args.use_bu == 'fixed':supp_infos.append('fixed_bu_feat')
    elif args.use_bu == 'adaptive':supp_infos.append('adaptive_bu_feat')
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
                                  dataset_name=args.dataset,caption_vocab=caption_vocab,data_dir=opt['data_dir'], use_bu='fixed', device=device)
    elif args.model_type == 'AoASpatial':
        model = AoASpatial_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                               dataset_name=args.dataset, caption_vocab=caption_vocab, device=device)
    elif args.model_type == 'AoADetection':
        model = AoADetection_Eng(model_settings_json=os.path.join(args.model_config_root, args.model_type + '.json'),
                                 dataset_name=args.dataset, caption_vocab=caption_vocab, data_dir=opt['data_dir'], use_bu=args.use_bu, device=device)
    print('model construction complete.')
    #print(model.model)  # you could choose to visualize the detailed model structure
    #---------------------------Operations---------------------------------------#
    if args.operation == 'train':
        train_dataloader = get_train_dataloader(args=args,opt=opt,caption_vocab=caption_vocab,supp_infos=supp_infos)
        # Note that Flickr8K/30K/COCO14KS contains 'val' and 'test' split, we only evaluate on the 'val' split after each training epoch
        # and we do not use beam search when evaluating on val split
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val',use_beam=False,supp_infos=supp_infos)
        # Please refer to the argparse_part for the function description of some parameters
        model.training(
            start_from=args.start_from,
            num_epochs=args.num_epochs,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],      # evaluation needs the raw captions stored in the json_file
            optimizer_type=args.optimizer,
            lm_rate=args.label_smoothing,                   # the smoothing rate for label_smoothing
            lr_opts={'learning_rate':args.learning_rate,                    # learning rate & lr decay settings
                     'cnn_finetune_learning_rate':args.cnn_finetune_learning_rate,      # only valid for models using cnn extractors like ResNet-101(NIC,BUTDSpatial,AoASpatial)
                     'cnn_finetune_start':args.cnn_finetune_start,
                     'lr_dec_start_epoch':args.learning_rate_decay_start,
                     'lr_dec_every':args.learning_rate_decay_every,
                     'lr_dec_rate':args.learning_rate_decay_rate},
            ss_opts={'ss_start_epoch':args.scheduled_sampling_start,        # scheduled sampling settings
                     'ss_inc_every':args.scheduled_sampling_increase_every,
                     'ss_inc_prob':args.scheduled_sampling_increase_prob,
                     'ss_max_prob':args.scheduled_sampling_max_prob},
            eval_beam_size=-1,                 # whether to eval with beam search strategy(we do not use beam search when evaluating during training)
            tqdm_visible=args.tqdm_visible              # enable the tqdm_bar to show the training/evaluation process
        )
    if args.operation == 'scst_train':
        # The operation is basically the same as the above, except to change the train_dataloader for SCST
        scst_train_dataloader = get_scst_train_dataloader(args=args,opt=opt,supp_infos=supp_infos)
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split='val',use_beam=False,supp_infos=supp_infos)
        # Parameter settings similar to the model training part
        # Note that we default the cnn_FT_start=True when start scst training since the models have been pretrained
        model.SCSTtraining(
            scst_num_epochs=args.scst_num_epochs,
            train_dataloader=scst_train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_caption_path=opt['val_caption_path'],
            optimizer_type=args.optimizer,
            scst_lr=args.scst_learning_rate,
            scst_cnn_FT_lr=args.scst_cnn_finetune_learning_rate,
            eval_beam_size=-1,
            start_from=args.start_from,
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'eval':
        # As mentioned before, you can choose to evaluate on the 'val' or 'test' split for Flickr/COCO14KS dataset
        # Note that we do not have the annotations for COCO17 test split, thus evaluation of COCO17 test split is not available.
        if args.eval_beam_size != -1:use_beam=True
        else:use_beam = False
        eval_dataloader = get_eval_dataloader(args=args,opt=opt,eval_split=args.eval_split,use_beam=use_beam,supp_infos=supp_infos)
        if args.eval_split == 'val':eval_caption_path = opt['val_caption_path']
        else:eval_caption_path = opt['test_caption_path']
        model.eval(
            dataset=args.dataset,
            split=args.eval_split,
            eval_scst=args.eval_scst,           # decide whether to evaluate models trained under scst optimization
            eval_best=args.eval_best,
            eval_dataloader=eval_dataloader,
            eval_caption_path=eval_caption_path,
            eval_beam_size=args.eval_beam_size, # decide whether to use beam search strategy when evaluating
            output_statics=False,               # when this option is set True, specific cider score results will be output
            tqdm_visible=args.tqdm_visible
        )
    if args.operation == 'sample':
        model.test(
            use_scst_model=args.eval_scst,
            use_best_model=args.eval_best,
            use_bu_feat=args.use_bu,
            img_root=args.COCO14_img_root,
            img_filename='COCO_val2014_000000356708.jpg',
            eval_beam_size=args.eval_beam_size
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #------------------------------------Global Settings------------------------------------#
    parser.add_argument('--dataset',type=str,default='COCO14',help='choose the dataset for training and evaluating')
    parser.add_argument('--model_type',type=str,default='NIC',help='choose the model_type, currently only supports two models')
    parser.add_argument('--dataset_config_root',type=str,default='./Configs/Datasets/',help='root to store dataset_configs')
    parser.add_argument('--model_config_root',type=str,default='./Configs/Models/',help='root to store model_configs')
    parser.add_argument('--gpu_id',type=str,default='0')
    parser.add_argument('--tqdm_visible',type=bool,default=True,help='choose to enable the tqdm_bar to show the training/evaluation process')
    parser.add_argument('--operation',type=str,default='train')

    #-----------------------------------Train Settings---------------------------------------#
    # Note that models trained w/w_o SCST algorithm are stored separately
    #------------------------------global training settings----------------------------------#
    parser.add_argument('--start_from',type=str,default='stratch',help='choose from "stratch" and "checkpoint",decide whether to train from stratch or checkpoints.')
    parser.add_argument('--img_size',type=int,default=224)
    parser.add_argument('--optimizer',type=str,default='Adam')
    #parser.add_argument('--use_preset',type=bool,default=True,help='when this option is set True, the model will be trained under the preset lr and optimizer in "MODELNAME.json"')
    parser.add_argument('--use_bu',type=str,default='unused',help='choose from "fixed","adaptive","unused", "fixed" means there are 36 bottom_up features per image, "adaptive" means there are 10~100 features per image')
    #-----------------------------XE Loss training settings----------------------------------#
    parser.add_argument('--num_epochs',type=int,default=30,help='maximum training epochs for training under XE Loss')
    parser.add_argument('--train_batch_size',type=int,default=128)
    parser.add_argument('--label_smoothing',type=float,default=0.1,help='use label smoothing for training. When set to 0.0 it is equal to CrossEntropyLoss')
    parser.add_argument('--learning_rate',type=float,default=4e-4)
    parser.add_argument('--cnn_finetune_learning_rate',type=float,default=1e-4,help='only valid for models using CNN to extract image features')
    parser.add_argument('--cnn_finetune_start',type=int,default=8,help='decide when to enable CNN finetune manually')
    parser.add_argument('--scheduled_sampling_start',type=int,default=0,help='when this option is set -1, scheduled sampling is disabled')
    parser.add_argument('--scheduled_sampling_increase_every',type=int,default=5)
    parser.add_argument('--scheduled_sampling_increase_prob',type=float,default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob',type=float,default=0.5)
    parser.add_argument('--learning_rate_decay_start',type=int,default=0,help='when this option is set -1,lr decay is disabled')
    parser.add_argument('--learning_rate_decay_every',type=int,default=3)
    parser.add_argument('--learning_rate_decay_rate',type=float,default=0.8)
    #--------------------------------SCST Training settings------------------------------------#
    parser.add_argument('--scst_num_epochs',type=int,default=50,help='maximum training epochs for training under SCST strategy')
    parser.add_argument('--scst_train_batch_size',type=int,default=128)
    parser.add_argument('--scst_learning_rate',type=float,default=1e-5)
    parser.add_argument('--scst_cnn_finetune_learning_rate',type=float,default=1e-5)

    #---------------------------------Evaluating Settings---------------------------------------#
    parser.add_argument('--eval_scst',type=bool,default=False,help='choose whether evaluating/testing on the model trained under SCST algorithm')
    parser.add_argument('--eval_best',type=bool,default=True,help='choose to eval/test with the best or resent pretrained model')
    parser.add_argument('--eval_split',type=str,default='test',help="since Flickr8K/30K/COCO14(KarpathySplitVer) contains test split, you can choose to eval on test split")
    parser.add_argument('--eval_batch_size',type=int,default=64)
    parser.add_argument('--eval_beam_size',type=int,default=3,help='when this option is set positive(e.g. =3), the beam search sampling is enabled during evaluating or testing')

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