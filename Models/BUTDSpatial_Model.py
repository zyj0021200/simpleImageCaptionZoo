import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as Fun
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.weight_norm import weight_norm

class EncoderCNN(nn.Module):
    def __init__(self,encoded_img_size=7):
        super(EncoderCNN,self).__init__()
        self.enc_img_size = encoded_img_size
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_img_size,encoded_img_size))
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,images):
        features = self.feature_extractor(images)   #(bsize,2048,H/32,W/32)
        features = self.adaptive_pool(features)    #(bsize,2048,7,7)
        bsize = images.size(0)
        num_pixels = features.size(2) * features.size(3)
        num_channels = features.size(1)
        global_features = self.GAP(features).view(bsize,-1) #(bsize,2048)
        features = features.permute(0,2,3,1)    #(bsize,7,7,2048)
        features = features.view(bsize,num_pixels,num_channels)     #(bsize,49,2048)
        return features,global_features

class SoftAttention(nn.Module):
    def __init__(self,enc_dim,hidden_dim,atten_dim,dropout=0.5):
        super(SoftAttention,self).__init__()
        self.enc_att = weight_norm(nn.Linear(enc_dim,atten_dim))
        self.dec_att = weight_norm(nn.Linear(hidden_dim,atten_dim))
        self.affine = weight_norm(nn.Linear(atten_dim,1))
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self,enc_features,dec_hidden):
        '''
        typical concat attention
        :param enc_features:
        :param dec_hidden:
        :return:
        '''
        enc_ctx = self.enc_att(enc_features)    #(bsize_t,num_pixels,atten_dim)
        dec_ctx = self.dec_att(dec_hidden)      #(bsize_t,atten_dim)
        atten = self.affine(self.dropout(self.relu(enc_ctx + dec_ctx.unsqueeze(1)))).squeeze(2)  #(bsize_t,num_pixels)
        alpha = torch.softmax(atten,dim=1)
        atten_weighted_enc = (enc_features * alpha.unsqueeze(2)).sum(dim=1)  #(bsize_t,num_pixels,2048)->(bsize_t,2048)
        return atten_weighted_enc,alpha

class DecoderRNN(nn.Module):
    def __init__(self,atten_dim,embed_dim,hidden_dim,vocab_size,enc_dim=2048,dropout=0.5,device='cpu'):
        super(DecoderRNN,self).__init__()
        self.atten_dim = atten_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.enc_dim = enc_dim
        self.device = device
        self.dropout = nn.Dropout(p=dropout)

        self.atten = SoftAttention(enc_dim=enc_dim,hidden_dim=hidden_dim,atten_dim=atten_dim)
        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim)
        self.TD_atten = nn.LSTMCell(input_size=embed_dim+enc_dim+hidden_dim,hidden_size=hidden_dim,bias=True)
        self.language_model = nn.LSTMCell(input_size=enc_dim+hidden_dim,hidden_size=hidden_dim,bias=True)
        self.predict = weight_norm(nn.Linear(in_features=hidden_dim,out_features=vocab_size))
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.predict.weight.data.uniform_(-0.1,0.1)
        self.predict.bias.data.fill_(0)

    def init_hidden_state(self,bsize):
        h = torch.zeros(bsize,self.hidden_dim).to(self.device)
        c = torch.zeros(bsize,self.hidden_dim).to(self.device)
        return h,c

    def forward(self,enc_features,global_features,captions,lengths):
        bsize = enc_features.size(0)
        num_pixels = enc_features.size(1)   #(bsize,49,2048)
        embeddings = self.embed(captions)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        predictions = torch.zeros(bsize,max(lengths),self.vocab_size).to(self.device)
        alphas = torch.zeros(bsize,max(lengths),num_pixels).to(self.device)     #atten weight for each pixel

        for time_step in range(max(lengths)):
            bsize_t = sum([l > time_step for l in lengths])
            h1,c1 = self.TD_atten(
                torch.cat([h2[:bsize_t],global_features[:bsize_t],embeddings[:bsize_t,time_step,:]],dim=1),
                (h1[:bsize_t],c1[:bsize_t])
            )
            atten_weighted_enc,alpha = self.atten(enc_features[:bsize_t],h1[:bsize_t])
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc[:bsize_t],h1[:bsize_t]],dim=1),
                (h2[:bsize_t],c2[:bsize_t])
            )
            preds = self.predict(self.dropout(h2))
            predictions[:bsize_t, time_step, :] = preds
            alphas[:bsize_t, time_step, :] = alpha

        pack_predictions = pack_padded_sequence(predictions, lengths, batch_first=True)
        return pack_predictions, alphas

    def sample(self,enc_features,global_features,max_len=20):
        bsize = enc_features.size(0)
        num_pixels = enc_features.size(1)   #(bsize,49,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        alphas = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2,global_features,embeddings],dim=1),
                (h1,c1)
            )
            atten_weighted_enc,alpha = self.atten(enc_features,h1)
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc,h1],dim=1),
                (h2,c2)
            )
            preds = self.predict(self.dropout(h2))
            pred_id = preds.max(1)[1]  # (bsize,)
            captions = pred_id.unsqueeze(1)  # (bsize,1)
            sampled_ids.append(captions)
            alphas.append(alpha.unsqueeze(1))  # (bsize,1,196)
        sampled_ids = torch.cat(sampled_ids, dim=1)  # (bsize,max_seq)
        alphas = torch.cat(alphas, dim=1)  # (bsize,max_seq,196)
        return sampled_ids, alphas

    def sample(self,enc_features,global_features,max_len=20):
        bsize = enc_features.size(0)
        num_pixels = enc_features.size(1)   #(bsize,49,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        captions = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        sampled_ids = []
        alphas = []
        for time_step in range(max_len):
            embeddings = self.embed(captions).squeeze(1)    #(bsize,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2,global_features,embeddings],dim=1),
                (h1,c1)
            )
            atten_weighted_enc,alpha = self.atten(enc_features,h1)
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc,h1],dim=1),
                (h2,c2)
            )
            preds = self.predict(self.dropout(h2))
            pred_id = preds.max(1)[1]  # (bsize,)
            captions = pred_id.unsqueeze(1)  # (bsize,1)
            sampled_ids.append(captions)
            alphas.append(alpha.unsqueeze(1))  # (bsize,1,196)
        sampled_ids = torch.cat(sampled_ids, dim=1)  # (bsize,max_seq)
        alphas = torch.cat(alphas, dim=1)  # (bsize,max_seq,196)
        return sampled_ids, alphas

    def sample_rl(self,enc_features,global_features,max_len=20):
        bsize = enc_features.size(0)
        num_pixels = enc_features.size(1)   #(bsize,49,2048)
        h1,c1 = self.init_hidden_state(bsize)
        h2,c2 = self.init_hidden_state(bsize)
        its = torch.LongTensor(bsize,1).fill_(1).to(self.device)
        seq = torch.zeros(bsize,max_len,dtype=torch.long).to(self.device)
        seqLogprobs = torch.zeros(bsize,max_len).to(self.device)
        for time_step in range(max_len):
            embeddings = self.embed(its).squeeze(1)    #(bsize,embed_dim)
            h1,c1 = self.TD_atten(
                torch.cat([h2,global_features,embeddings],dim=1),
                (h1,c1)
            )
            atten_weighted_enc,alpha = self.atten(enc_features,h1)
            h2,c2 = self.language_model(
                torch.cat([atten_weighted_enc,h1],dim=1),
                (h2,c2)
            )
            preds = self.predict(self.dropout(h2))
            logprobs = Fun.log_softmax(preds, dim=1)
            prob_prev = torch.exp(logprobs)
            its = torch.multinomial(prob_prev, num_samples=1)  # sample a word  #(bsize,1)
            sampleLogprobs = logprobs.gather(1, its)  # gather the logprobs at sampled positions   #(bsize,1)
            its = its.clone()
            if time_step == 0:
                unfinished = abs(its - 2) > 0
            else:
                unfinished = unfinished * (abs(its - 2) > 0)
            its = its * unfinished.type_as(its)
            seq[:, time_step] = its.view(-1)
            seqLogprobs[:, time_step] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0: break
        return seq,seqLogprobs

    def beam_search_sample(self,enc_features,global_features,beam_size=5):
        '''
        :param enc_features:(1,h*w,2048)
        :param global_features:(1,2048)
        :param beam_size:scalar
        :return:
        '''
        num_pixels = enc_features.size(1)
        k = beam_size
        k_prev_words = torch.LongTensor(k,1).fill_(1).to(self.device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k,1).to(self.device)
        enc_features = enc_features.expand(k,enc_features.shape[1],enc_features.shape[2])   #(k,h*w,2048)
        global_features = global_features.expand(k,global_features.shape[1])    #(k,2048)
        complete_seqs = list()
        complete_seqs_scores = list()
        alphas = torch.zeros(k, 1, num_pixels).to(self.device)
        complete_alphas = list()

        step = 1
        max_step_limit = 50
        h1,c1 = self.init_hidden_state(bsize=k)    #(k,hidden_dim)
        h2,c2 = self.init_hidden_state(bsize=k)
        while step<=max_step_limit:
            embeddings = self.embed(k_prev_words).squeeze(1)    #(s=active_beam_num,embed_dim)
            h1,c1 = self.TD_atten(torch.cat([h2,global_features,embeddings],dim=1),(h1,c1))
            atten_weighted_enc, alpha = self.atten(enc_features,h1)  #(s,2048),(s,196)

            h2,c2 = self.language_model(torch.cat((atten_weighted_enc,h1),dim=1),(h2,c2))     #(s,hidden_dim)

            scores = self.predict(self.dropout(h2))
            scores = Fun.log_softmax(scores,dim=1)
            scores = top_k_scores.expand_as(scores) + scores    #(s,vocab_size)
            if step == 1:
                top_k_scores,top_k_words = scores[0].topk(k,0,True,True) #(s)
            else:
                top_k_scores,top_k_words = scores.view(-1).topk(k,0,True,True)
            prev_word_inds = top_k_words / self.vocab_size
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)
            alphas = torch.cat([alphas[prev_word_inds],alpha.unsqueeze(1)],dim=1)   #(s,step+1,196)

            incomplete_inds = [ind for ind,next_word in enumerate(next_word_inds) if next_word != 2]
            complete_inds = list(set(range(len(next_word_inds)))-set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_alphas.extend(alphas[complete_inds].tolist())

            k -= len(complete_inds)
            if k==0:
                break

            seqs = seqs[incomplete_inds]
            enc_features = enc_features[prev_word_inds[incomplete_inds]]
            global_features = global_features[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_seqs)>0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            seq_tensor = torch.Tensor(seq).unsqueeze(0)
            complete_alpha = complete_alphas[i]
            alpha_tensor = torch.Tensor(complete_alpha).unsqueeze(0)    #(1,len,196)
        else:
            i = torch.max(top_k_scores,dim=0)[1].item()
            seq_tensor = seqs[i].unsqueeze(0)
            alpha_tensor = alphas[i].unsqueeze(0)

        return seq_tensor,alpha_tensor

class BUTDSpatial_Captioner(nn.Module):
    def __init__(self,encoded_img_size,atten_dim,embed_dim,hidden_dim,vocab_size,dropout=0.5,device='cpu'):
        super(BUTDSpatial_Captioner,self).__init__()
        self.encoder = EncoderCNN(encoded_img_size=encoded_img_size)
        self.decoder = DecoderRNN(atten_dim=atten_dim,embed_dim=embed_dim,hidden_dim=hidden_dim,vocab_size=vocab_size,dropout=dropout,device=device)
        self.cnn_fine_tune(flag=False)

    def cnn_fine_tune(self, flag=False):
        if flag:
            for module in list(self.encoder.feature_extractor.children())[7:]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            for params in self.encoder.feature_extractor.parameters():
                params.requires_grad = False

    def forward(self,images,captions,lengths):
        enc_features,global_features = self.encoder(images) #(bsize,49,2048)/(bsize,2048)
        packed_predictions,alphas = self.decoder(enc_features,global_features,captions,lengths)
        return packed_predictions

    def sampler(self,images,max_len=20):
        enc_features,global_features = self.encoder(images)
        sampled_ids,alphas = self.decoder.sample(enc_features,global_features,max_len)
        return sampled_ids

    def sampler_rl(self,images,max_len=20):
        enc_features,global_features = self.encoder(images)
        seq,seqLogprobs = self.decoder.sample_rl(enc_features=enc_features,global_features=global_features,max_len=max_len)
        return seq,seqLogprobs

    def beam_search_sampler(self,images,beam_size=5):
        enc_features,global_features = self.encoder(images)
        sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=enc_features,global_features=global_features,beam_size=beam_size)
        return sampled_ids

    def eval_test_image(self,image,caption_vocab,max_len=20,eval_beam_size=-1):
        assert image.size(0) == 1
        enc_features,global_features = self.encoder(image)
        if eval_beam_size != -1:
            sampled_ids,alphas = self.decoder.beam_search_sample(enc_features=enc_features,global_features=global_features,beam_size=eval_beam_size)
        else:
            sampled_ids,alphas = self.decoder.sample(enc_features=enc_features,global_features=global_features,max_len=max_len)
        caption_ids = sampled_ids[0].cpu().detach().numpy()
        caption = []
        for word_id in caption_ids:
            word = caption_vocab.ix2word[word_id]
            if word == '<end>':
                break
            elif word != '<sta>':
                caption.append(word)
        return caption,[alphas]