import torch
import torch.nn.functional as F
from torch import nn 
from xbert import BertConfig, BertForMaskedLM
from transformers import EsmForMaskedLM, EsmConfig

class proteinXVL(nn.Module):
    def __init__(self, 
                 protein_tokenizer = None,
                 smiles_tokenizer = None,
                 config = None,
                 temp = 0.07
                 ):
        super().__init__()

        embed_dim = config['embed_dim']
        self.protein_tokenizer = protein_tokenizer
        self.smiles_tokenizer = smiles_tokenizer

        protein_config = EsmConfig.from_json_file('./config_bert_protein_encoder.json')
        self.proteinEncoder = EsmForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
        
        smiles_config = BertConfig.from_json_file('./config_bert_smiles_and_fusion_encoder.json')
        self.smilesEncoder = BertForMaskedLM(config = smiles_config)

        smilesWidth = self.smilesEncoder.config.hidden_size
        proteinWidth = self.proteinEncoder.config.hidden_size

        self.smilesProj = nn.Linear(smilesWidth, embed_dim)
        self.proteinProj = nn.Linear(proteinWidth, embed_dim)

        self.affinity_reg = nn.Sequential(nn.Linear(proteinWidth+smilesWidth, (proteinWidth+smilesWidth)*0.5))
        
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.queue_size = config['queque_size']
        self.momentum = config['momentum']

        ### Momentum Model ###
        self.smilesEncoder_m = BertForMaskedLM(config = smiles_config)
        self.proteinEncoder_m = EsmForMaskedLM(config = protein_config)
        
        self.smilesProj_m = nn.Linear(smilesWidth, embed_dim)
        self.proteinProj_m = nn.Linear(proteinWidth, embed_dim)
        
        self.model_pairs = [[self.proteinEncoder, self.proteinEncoder_m],
                            [self.proteinProj, self.proteinProj_m],
                            [self.smilesEncoder, self.smilesEncoder_m],
                            [self.smilesProj, self.smilesProj_m]]
        
        self.copy_params()

        #Create queue
        self.register_buffer("protein_queue", torch.randn(embed_dim, self.queqeq_size))
        self.register_buffer("smiles_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.protein_queue = nn.functional.normalize(self.protein_queue, dim=0)
        self.smiles_queue = nn.functional.normalize(self.smiles_queue, dim=0)
        
    def forward(self, proteinIds, smilesIds, protAttentionMask, smilesAttentionMask, alpha):

        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)
        
        #1 Protein Embedding
        encProtein = self.proteinEncoder(proteinIds, encoder_attention_mask=protAttentionMask, return_dict=True).last_hidden_state
        proteinFeats= F.normalize(self.proteinProj(encProtein[:,0,:]), dim=-1)

        #2 SMILES Embedding
        encSMILES = self.smilesEncoder(smilesIds, encoder_attention_mask=smilesAttentionMask, retrun_dict=True)
        smilesFeats = F.normalize(self.smilesProj(encSMILES[:,0,:]), dim=-1)

        #3 Contrastive Loss

        with torch.no_grad():
            self._momentum_update()
            
            encProtein_m = self.proteinEncoder_m(proteinIds, encoder_attention_mask=protAttentionMask, return_dict=True).last_hidden_state
            proteinFeats_m = F.normalize(self.proteinProj_m(encProtein_m[:,0,:]), dim=-1)
            proteinFeatsAll = torch.cat([proteinFeats_m.t(), self.protein_queue.clone().detach()], dim=1)

            encSMILES_m = self.smilesEncoder_m(smilesIds, encoder_attention_mask=smilesAttentionMask, return_dict=True).last_hidden_state
            smilesFeats_m = F.normalize(self.smilesProj_m(encSMILES_m[:,0,:]), dim=-1)
            smilesFeatsAll = torch.cat([smilesFeats_m.t(), self.protein_queue.clone().detach()], dim=1)

            sim_p2s_m = proteinFeats_m @ smilesFeatsAll / self.temp
            sim_s2p_m = smilesFeats_m @ proteinFeatsAll / self.temp
            sim_p2p_m = proteinFeats_m @ proteinFeatsAll / self.temp
            sim_s2s_m = smilesFeats_m @ smilesFeatsAll / self.temp

            sim_targets_diff = torch.zeros(sim_p2s_m.size()).to(proteinIds.device)
            sim_targets_diff.fill_diagonal_(1)
            sim_targets_same = torch.zeros(sim_p2p_m.size()).to(proteinIds.device)
            sim_targets_same.fill_diagonal_(1)
          
            sim_p2s_targets = alpha * F.softmax(sim_p2s_m, dim=1) + (1-alpha) * sim_targets_diff
            sim_s2p_targets = alpha * F.softmax(sim_s2p_m, dim=1) + (1-alpha) * sim_targets_diff
            sim_p2p_targets = alpha * F.softmax(sim_p2p_m, dim=1) + (1-alpha) * sim_targets_same
            sim_s2s_targets = alpha * F.softmax(sim_s2s_m, dim=1) + (1-alpha) * sim_targets_same

        sim_p2s = proteinFeats @ smilesFeatsAll / self.temp
        sim_s2p = smilesFeats @ proteinFeatsAll / self.temp
        sim_p2p = proteinFeats @ proteinFeatsAll / self.temp
        sim_s2s = smilesFeats @ smilesFeatsAll / self.temp

        loss_p2s = -torch.sum(F.log_softmax(sim_p2s, dim=1)*sim_p2s_targets, dim=1).mean()
        loss_s2p = -torch.sum(F.log_softmax(sim_s2p, dim=1)*sim_s2p_targets, dim=1).mean()
        loss_p2p = -torch.sum(F.log_softmax(sim_p2p, dim=1)*sim_p2p_targets, dim=1).mean()
        loss_s2s = -torch.sum(F.log_softmax(sim_s2s, dim=1)*sim_s2s_targets, dim=1).mean()

        contrastive_loss = (loss_p2s + loss_s2p + loss_p2p + loss_s2s) * 0.5

        self._dequeue_and_enqueue(proteinFeats_m, smilesFeats_m)

        # Regression(Binding Affinity) Loss
        

        regression_loss = None
        return contrastive_loss + regression_loss


 
            
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, property_feat, smiles_feat):

        property_feats = property_feat
        smiles_feats = smiles_feat

        batch_size = property_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.property_queue[:, ptr:ptr + batch_size] = property_feats.T
        self.smiles_queue[:, ptr:ptr + batch_size] = smiles_feats.T
        ptr = (ptr + batch_size) % self.queue_size 

        self.queue_ptr[0] = ptr