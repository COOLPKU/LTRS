import torch

from utils import *
from transformers import BertTokenizer, BertConfig, BertModel

class GlossEncoder(torch.nn.Module):
    def __init__(self, encoder_name):
        super(GlossEncoder, self).__init__()
        self.gloss_encoder= BertModel.from_pretrained(encoder_name)
        self.gloss_hdim = 768
    def forward(self, input_ids, attention_mask,token_type_ids):

        gloss_output = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0][:, 0]
        gloss_output = torch.nn.functional.normalize(gloss_output, p=2, dim=1)
        return gloss_output

class ContextEncoder(torch.nn.Module):
    def __init__(self, encoder_name,device):
        super(ContextEncoder, self).__init__()
        self.context_encoder= BertModel.from_pretrained(encoder_name)
        self.context_hdim=768
        self.device=device

    def forward(self, input_ids, attention_mask,token_type_ids,indices_list):
        context_output = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0]


        output_data = []
        for one_ind, t in zip(indices_list, torch.split(context_output, 1, dim=0)):
            indice=torch.tensor(one_ind,dtype=torch.long).to(self.device)
            temp = torch.index_select(t.squeeze(), 0, indice)
            output_data.append(torch.mean(temp, dim=0))

        output_data = torch.stack(output_data)

        output_data = torch.nn.functional.normalize(output_data, p=2, dim=1)

        return output_data



class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_name,device='cuda:0'):
        super(BiEncoderModel, self).__init__()
        self.context_encoder = ContextEncoder(encoder_name,device=device)
        self.gloss_encoder = GlossEncoder(encoder_name)
        assert self.context_encoder.context_hdim == self.gloss_encoder.gloss_hdim

    def context_forward(self, input_ids, attention_mask,token_type_ids,indices_list):
        return self.context_encoder.forward( input_ids, attention_mask,token_type_ids,indices_list)

    def gloss_forward(self, gloss_input):
        return self.gloss_encoder.forward(**gloss_input)
