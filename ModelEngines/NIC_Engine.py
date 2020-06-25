from Engine import Engine

class NIC_Eng(Engine):
    def get_model_params(self):
        cnn_extractor_params = list(filter(lambda p: p.requires_grad, self.model.encoder.feature_extractor.parameters()))
        captioner_params = list(self.model.encoder.img_embedding.parameters()) + \
                           list(self.model.encoder.bn.parameters()) + \
                           list(self.model.decoder.parameters())
        return cnn_extractor_params,captioner_params