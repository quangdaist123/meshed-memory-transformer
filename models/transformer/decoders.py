import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.transformer.attention import MultiHeadAttention
from models.transformer.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.containers import Module, ModuleList


class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad
        enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
        enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1)))
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class MeshedDecoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoder, self).__init__()
        self.d_model = d_model
        # self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.word_emb = AutoModel.from_pretrained("vinai/phobert-base")
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        if isinstance(input, tuple):
            temp = list(input)
            for i in range(len(input)):
                if temp[i]:
                    temp[i] = torch.tensor([self.tokenizer.encode(input[i])])
                else:
                    temp[i] = torch.tensor([[99999]])
            max_len = 0
            for line in temp:
                if line.shape[1] > max_len:
                    max_len = line.shape[1]

            padding_mask = []
            for i in range(len(temp)):
                is_none_input = (temp[i] == torch.tensor([[99999]])).sum()
                if is_none_input:
                    print("Bammmmmmmmmmmmm")
                    pad_num = int(max_len - 2)
                    pad_token = torch.tensor([[1] * pad_num], dtype=torch.int32)
                    temp[i] = torch.cat((torch.tensor([[0, 2]]), pad_token), 1)
                    unpadded = torch.tensor([[1] * 2], dtype=torch.int32)
                    padded = torch.tensor([[0] * (max_len - 2)], dtype=torch.int32)
                    padding_mask.append(torch.cat((unpadded, padded), 1))

                else:
                    seq_len = temp[i].shape[1]
                    pad_num = int(max_len - seq_len)
                    pad_token = torch.tensor([[1] * pad_num], dtype=torch.int32)
                    temp[i] = torch.cat((temp[i], pad_token), 1)
                    # attention mask
                    unpadded = torch.tensor([[1] * seq_len], dtype=torch.int32)
                    padded = torch.tensor([[0] * (max_len - seq_len)], dtype=torch.int32)
                    padding_mask.append(torch.cat((unpadded, padded), 1))
            padding_mask = torch.cat(padding_mask, 0)
            input = torch.cat(temp, 0)
            input, padding_mask = input.to("cuda"), padding_mask.to("cuda")

        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        if isinstance(input, tuple):
            out = self.word_emb(input, attention_mask=padding_mask).last_hidden_state + self.pos_emb(seq)
        else:
            out = self.word_emb(input).last_hidden_state + self.pos_emb(seq)
        # # Đảo lại giá trị bool trong mask_self_attn để thay thế padding_mask
        # # Đã cùng shape
        # print("mask_self_attention", mask_self_attention[:, -1, -1, :])
        # print("mask_self_attention", mask_self_attention[:, -1, -1, :].shape)
        # print("padding_mask", padding_mask)
        # print("padding_mask", padding_mask.shape)
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
