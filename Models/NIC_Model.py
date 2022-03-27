import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as Fun
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self,embed_dim):
        super(EncoderCNN,self).__init__()
        self.embed_dim = embed_dim
        resnet = models.resnet101(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.pool = resnet.avgpool
        self.img_embedding = weight_norm(nn.Linear(resnet.fc.in_features,embed_dim))
        self.bn = nn.BatchNorm1d(embed_dim,momentum=0.01)

    def forward(self, images):
        '''
        embed raw image tensors into visual features
        :param images: (bsize,3,224,224)
        :return: embed_features: (bsize,embed_dim)
        '''
        features = self.feature_extractor(images)   #(bsize,3,224,224) -> (bsize,2048,7,7)
        features = self.pool(features)      #(bsize,2048,7,7)->(bsize,2048,1,1)
        features = features.view(features.size(0),-1)
        embed_features = self.img_embedding(features)   #(bsize,2048)->(bsize,embed_dim)
        return embed_features

class DecoderRNN(nn.Module):
    def __init__(self,embed_dim,hidden_dim,vocab_size,dropout=0.5,device='cpu'):
        super(DecoderRNN,self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device
        self.ss_prob = 0.0
        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.lstm = nn.LSTMCell(input_size=embed_dim,hidden_size=hidden_dim,bias=True)
        self.predict = weight_norm(nn.Linear(in_features=hidden_dim,out_features=vocab_size,bias=True))
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self,bsize,hidden_dim,features):
        h = torch.zeros(bsize,hidden_dim).to(self.device)
        c = torch.zeros(bsize,hidden_dim).to(self.device)
        h,c = self.lstm(features,(h,c))
        return h,c

    def forward(self,features,captions,lengths):
        '''
        XE Loss training process.
        :param features: (bsize,embed_dim)
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor, e.g.:
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_predictions: packed_padded_sequence:((total_tokens_nums,vocab_size),indices(could be ignored,not used in training))
        '''
        bsize = features.size(0)
        h,c = self.init_hidden_state(bsize=bsize,hidden_dim=self.hidden_dim,features=features)
        #embeddings = self.embed(captions)   #(bsize,max_len,embed_dim)
        predictions = torch.zeros(bsize,max(lengths),self.vocab_size).to(self.device)

        for time_step in range(captions.size(1)-1):
            bsize_t = sum([caption_len > time_step for caption_len in lengths])
            if time_step >= 2 and self.ss_prob > 0.0:
                sample_prob = torch.zeros(bsize_t).uniform_(0,1).to(self.device)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:bsize_t,time_step]
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = captions[:bsize_t,time_step].clone()
                    prob_prev = torch.softmax(preds,dim=1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = captions[:bsize_t,time_step]

            embeddings = self.embed(it) #(bsize_t,embed_dim)
            h,c = self.lstm(embeddings,(h[:bsize_t],c[:bsize_t]))
            preds = self.predict(self.dropout(h))   #(bsize,vocab_size)
            predictions[:bsize_t,time_step,:] = preds

        pack_predictions = pack_padded_sequence(predictions,lengths,batch_first=True)
        return pack_predictions

    def sample(self,features,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param features: (bsize,embed_dim)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        bsize = features.size(0)
        h,c = self.init_hidden_state(bsize=bsize,hidden_dim=self.hidden_dim,features=features)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h,c = self.lstm(embeddings,(h,c))
            preds = self.predict(self.dropout(h))
            pred_id = preds.max(1)[1]   #(bsize,)
            captions = pred_id.unsqueeze(1) #(bsize,1)
            sampled_ids.append(captions)
        sampled_ids = torch.cat(sampled_ids,dim=1)  #(bsize,max_seq)
        return sampled_ids

    def sample_rl(self,features,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param features: (bsize,embed_dim)
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        bsize = features.size(0)
        h,c = self.init_hidden_state(bsize=bsize,hidden_dim=self.hidden_dim,features=features)
        its = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        seq = torch.zeros(bsize,max_len,dtype=torch.long).to(self.device)
        seqLogprobs = torch.zeros(bsize,max_len).to(self.device)
        for time_step in range(max_len):
            embeddings = self.embed(its).squeeze(1)    #(bsize,embed_dim)
            h,c = self.lstm(embeddings,(h,c))
            preds = self.predict(self.dropout(h))   #(bsize,vocab_size)
            logprobs = Fun.log_softmax(preds,dim=1)
            prob_prev = torch.exp(logprobs)
            its = torch.multinomial(prob_prev,num_samples=1)     #sample a word  #(bsize,1)
            sampleLogprobs = logprobs.gather(1,its)      #gather the logprobs at sampled positions   #(bsize,1)
            its = its.clone()
            if time_step == 0:
                unfinished = abs(its-2) > 0
            else:
                unfinished = unfinished * (abs(its-2) > 0)
            its = its * unfinished.type_as(its)
            seq[:, time_step] = its.view(-1)
            seqLogprobs[:, time_step] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0: break
        return seq,seqLogprobs

    def beam_search_sample(self,features,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param features: (bsize=1,embed_dim)
        :param beam_size: beam numbers scalar
        :return:seq_tensor:(1,sentence_len)
        '''
        k = beam_size
        k_prev_words = torch.LongTensor(k,1).fill_(1).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k,1).to(self.device)
        features = features.expand(k,features.shape[1])
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        max_step_limit = 50
        h,c = self.init_hidden_state(bsize=k,hidden_dim=self.hidden_dim,features=features)
        while step<=max_step_limit:
            embeddings = self.embed(k_prev_words).squeeze(1)    #(s,embed_dim)
            h,c = self.lstm(embeddings,(h,c))
            scores = self.predict(self.dropout(h))
            scores = Fun.log_softmax(scores,dim=1)
            scores = top_k_scores.expand_as(scores) + scores    #(s,vocab_size)
            if step == 1:
                top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #(s)
            else:
                top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True)
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)

            incomplete_inds = [ind for ind,next_word in enumerate(next_word_inds) if next_word != 2]
            complete_inds = list(set(range(len(next_word_inds)))-set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)
            if k==0:
                break

            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_seqs)>0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            seq_tensor = torch.Tensor(seq).unsqueeze(0)
        else:
            i = torch.max(top_k_scores,dim=0)[1].item()
            seq_tensor = seqs[i].unsqueeze(0)

        return seq_tensor

class NIC_Captioner(nn.Module):
    def __init__(self,embed_dim,hidden_dim,vocab_size,dropout=0.5,device='cpu'):
        super(NIC_Captioner,self).__init__()
        self.encoder = EncoderCNN(embed_dim=embed_dim)
        self.decoder = DecoderRNN(embed_dim=embed_dim,hidden_dim=hidden_dim,vocab_size=vocab_size,dropout=dropout,device=device)
        self.cnn_finetune(flag=False)

    def get_param_groups(self,lr_dict):
        cnn_extractor_params = list(filter(lambda p: p.requires_grad, self.encoder.feature_extractor.parameters()))
        captioner_params = list(self.encoder.img_embedding.parameters()) + \
                           list(self.encoder.bn.parameters()) + \
                           list(self.decoder.parameters())
        assert lr_dict.__contains__('cnn_ft_lr')
        param_groups = [
            {'params':captioner_params,'lr':lr_dict['lr']},
            {'params':cnn_extractor_params,'lr':lr_dict['cnn_ft_lr']}
        ]
        return param_groups

    def cnn_finetune(self,flag=False):
        '''
        we only fine tune on the last layer of resnet-101
        :param flag: enable cnn fine_tune when set true.
        '''
        if flag:
            for module in list(self.encoder.feature_extractor.children())[7:]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for params in self.encoder.feature_extractor.parameters():
                params.requires_grad = False

    def forward(self,visual_inputs,captions,lengths):
        '''
        XE Loss training process.
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using NIC model, only raw img tensors are required
        :param captions: (bsize,max_len) sorted in length,torch.LongTensor, e.g.:
                [[1(<sta>),24,65,3633,54,234,67,34,12,2(<end>)],    #max_len=10
                [1(<sta>),45,5434,235,12,58,11,2(<end>),0,0],
                ...
                [1(<sta>),24,7534,523,12,2(<end>),0,0,0,0]]
        :param lengths:[9,7,...,5]
                #note that length[i] = len(caption[i])-1 since we skip the 2<end> token when feeding the captions-tensor into the model
                and we skip the 1<sta> token when generating predictions for loss calculation. Thus the total training step in this batch equals to max(lengths)
        :return:packed_predictions: packed_padded_sequence:((total_tokens_nums,vocab_size),indices(could be ignored,not used in training))
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        embed_features = self.encoder(images)   #(bsize,embed_dim)
        packed_predictions = self.decoder(embed_features,captions,lengths)
        return packed_predictions

    def sampler(self,visual_inputs,max_len=20):
        '''
        Use Greedy-search to generate predicted captions
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using NIC model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:sampled_ids:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        embed_features = self.encoder(images)   #(bsize,embed_dim)
        sampled_ids = self.decoder.sample(features=embed_features,max_len=max_len)
        return sampled_ids

    def sampler_rl(self,visual_inputs,max_len=20):
        '''
        Use Monte Carlo method(Random Sampling) to generate sampled predictions for scst training
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using NIC model, only raw img tensors are required
        :param max_len: maximum length(or time_step) of the generated sentence
        :return:seq:(bsize,max_len)
                seqLogprobs:(bsize,max_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize,3,224,224)
        embed_features = self.encoder(images)   #(bsize,embed_dim)
        seq,seqLogprobs = self.decoder.sample_rl(features=embed_features,max_len=max_len)
        return seq,seqLogprobs

    def beam_search_sampler(self,visual_inputs,beam_size=5):
        '''
        Use Beam search method to generate predicted captions, asserting bsize=1
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize,3,224,224)}
            dict of available visual features. when using NIC model, only raw img tensors are required
        :param beam_size: beam numbers scalar
        :return: sampled_ids:(bsize=1,sentence_len)
        '''
        images = visual_inputs['img_tensors']   #(bsize=1,3,224,224)
        embed_features = self.encoder(images)   #(bsize=1,embed_dim)
        sampled_ids = self.decoder.beam_search_sample(features=embed_features,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,visual_inputs,caption_vocab,max_len=20,eval_beam_size=-1):
        '''
        Tests on single given image.
        :param visual_inputs: {'img_tensors':torch.FloatTensor(bsize=1,3,224,224)}
            dict of available visual features. when using NIC model, only raw img tensors are required
        :param caption_vocab: pkl-file. used to translate the generated sentence into human language.
        :param max_len: maximum length(or time_step) of the generated sentence
        :param eval_beam_size: beam numbers scalar
        :return:caption: generated caption for given image.
                additional output: e.g. attention weights over different visual features at each time step during training.(not used in NIC)
        '''
        image = visual_inputs['img_tensors']    #(bsize=1,3,224,224)
        assert image.size(0) == 1
        embed_features = self.encoder(image)
        if eval_beam_size!=-1:
            sampled_ids = self.decoder.beam_search_sample(features=embed_features,beam_size=eval_beam_size)
        else:
            sampled_ids = self.decoder.sample(features=embed_features,max_len=max_len)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[]       #for additional_output
