from Engine import Engine
import torch
import numpy as np
from Utils import visualize_att,visualize_att_bboxes

#--------------------Models using ResNet to extract image features---------------------#
class BUTDSpatial_Eng(Engine):

    def show_additional_rlt(self,additional_outputs,visual_inputs,image,caption):
        alphas = additional_outputs[0]
        alphas = alphas.squeeze(0).cpu()  #(1,max_len,49)->(max_len,49)
        sta_alpha = torch.zeros(1,alphas.size(1))   #time_step_0 has no alpha value, just for convience of visualization
        alphas = torch.cat((sta_alpha,alphas),dim=0)    #(max_len+1,49)
        alphas = alphas.view(alphas.size(0),self.settings['enc_img_size'],self.settings['enc_img_size'])     #(max_len,14,14)/(max_len,7,7)
        alphas = alphas.detach().numpy()
        caption.insert(0,'<sta>')
        caption.append('<end>')
        visualize_att(image=image,alphas=alphas,caption=caption)

#---------------------Models using pretrained Bottom-up features------------------------#
class BUTDDetection_Eng(Engine):

    def modify_visual_inputs(self,img_tensors,supp_info_datas=None):
        '''
        :param img_tensors: (bsize,3,224,224)
        :param supp_info_datas:({'bu_feat':array(...),'bu_bbox':array(...)},...,{'bu_feat':array(...),'bu_bbox':array(...)})
                bu_feat:(36,2048) np.ndarry bu_bbox:(36,4) np.ndarry
        :return:inputs:{'bu_feats','bu_bboxes','bu_masks'}
        '''
        bu_feats = []
        bu_bboxes = []
        for supp_info_data in supp_info_datas:
            bu_feats.append(supp_info_data['bu_feat'])
            bu_bboxes.append(supp_info_data['bu_bbox'])
        max_bu_len =max([_.shape[0] for _ in bu_feats])      #record the maximum numbers of bu_features the images in this batch has
        reshaped_bu_feats =np.zeros(shape=(len(supp_info_datas),max_bu_len,bu_feats[0].shape[1]),dtype='float32') #(bsize,max_bu_len,2048)
        bu_masks = np.zeros(reshaped_bu_feats.shape[:2],dtype='float32')      # (bsize,max_bu_len)
        for i in range(len(supp_info_datas)):
            reshaped_bu_feats[i,:bu_feats[i].shape[0]] = bu_feats[i]   # fill in the bu_features in the reshaped numpy.ndarry
            bu_masks[i,:bu_feats[i].shape[0]] = 1
        if bu_masks.sum() == bu_masks.size:
            bu_masks = None                          # when using fixed bu_features, mask is set None
        else:
            bu_masks = torch.from_numpy(bu_masks).float().to(self.device)
        reshaped_bu_feats = torch.from_numpy(reshaped_bu_feats).float().to(self.device)
        inputs = {'bu_feats':reshaped_bu_feats,'bu_bboxes':bu_bboxes,'bu_masks':bu_masks}
        return inputs

    def show_additional_rlt(self,additional_outputs,visual_inputs,image,caption):
        alphas = additional_outputs[0]
        alphas = alphas.squeeze(0)  #(1,max_len,36)->(max_len,36)
        alphas = alphas.cpu()
        sta_alpha = torch.zeros(1,alphas.size(1))   #time_step_0 has no alpha value, just for convience of visualization
        alphas = torch.cat((sta_alpha,alphas),dim=0)
        alphas = alphas.detach().numpy()
        bboxes = visual_inputs['bu_bboxes'][0]  #(36,4)
        caption.insert(0,'<sta>')
        caption.append('<end>')
        visualize_att_bboxes(image=image,alphas=alphas,bboxes=bboxes,caption=caption)
