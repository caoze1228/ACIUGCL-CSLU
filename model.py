from typing import Literal

import torch
from torch import nn
from snoop import snoop
from torch.nn import functional as tf
import dgl
from params import Params
from dgl import nn as dglnn
from utils import make_state_graph, move_dgl_to_cuda
from transformers import BertModel
import inspect


def device_watch(source, value):
    if hasattr(value, 'device'):
        return f"{source}.device", getattr(value, 'device')
    else:
        return f"{source} = ", "no device"


snoop.install(watch_extras=[device_watch])
BERT_DIM = 768


class IntentClassifier(nn.Module):
    def __init__(self, in_dim, num_classes: int, dropout=0.5):
        super(IntentClassifier, self).__init__()
        self.ffn = nn.Sequential(
                nn.Linear(in_dim, num_classes),
                nn.Dropout(dropout)
        )

    def forward(self, e_0, e_L):
        """
        :param e_0: [T, dim]
        :param e_L: [T, dim]
        :return:
        """
        e_init = e_0[-1, :]
        e_L_t = e_L[-1, :]
        return self.ffn.forward(torch.cat([e_init, e_L_t], dim=-1))


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, device, num_layers=2):
        super().__init__()

        layers = []
        # two-layer GCN
        for _ in range(num_layers):
            layers.append(
                    dglnn.GraphConv(in_size, hid_size, activation=tf.relu).to(device)
            )
        layers.append(dglnn.GraphConv(hid_size, out_size).to(device))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.5)

    # @snoop
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class SlotLSTMDecoder(nn.Module):
    def __init__(self, num_slots: int, input_feats: int, num_intents: int = 0, dropout=0.5,
                 slot_embed_dim: int = 10):
        super(SlotLSTMDecoder, self).__init__()
        self.lstm_input_dim = num_intents + input_feats + slot_embed_dim
        self.slot_embedding_layer = nn.Embedding(num_embeddings=num_slots,
                                                 embedding_dim=slot_embed_dim)
        if num_intents != 0:
            self.intent_embedding = nn.Embedding(num_embeddings=num_intents,
                                                 embedding_dim=num_intents)
            self.intent_embedding.weight.data = torch.eye(num_intents)
            self.intent_embedding.weight.requires_grad = False
        else:
            self.intent_embedding = None
        self.__init_tensor = nn.Parameter(
                torch.randn(1, slot_embed_dim), requires_grad=True
        )
        self.dropout = nn.Dropout(0.5)
        self.lstm_layer = nn.LSTM(
                input_size=self.lstm_input_dim,
                hidden_size=num_slots,
                bidirectional=False,
                batch_first=True,
                num_layers=1,
                dropout=dropout
        )
        self.out_linear = nn.Linear(
                in_features=num_slots,
                out_features=num_slots,
                bias=False
        )
        self.lstm_cell = nn.LSTMCell(
                input_size=self.lstm_input_dim,
                hidden_size=num_slots
        )

    # @snoop
    def forward(self, encoder_out: torch.Tensor, intent_out: torch.Tensor, end_idx=-1):
        """

        :param encoder_out: [L, DIM]
        :param intent_out: idx
        :param end_idx: to-deal with padding
        :return: [L, K]
        """
        seq_len = encoder_out.size()[0]
        output_tensor_list, sent_start_pos = [], 0
        if self.intent_embedding is not None:
            eidx = end_idx if end_idx != -1 else seq_len
            seg_hiddens = encoder_out[sent_start_pos: eidx, :]
            intent_hidden = self.intent_embedding.forward(intent_out).view(1, -1)
            prev_state = self.__init_tensor
            prev_hx = None
            prev_cx = None
            all_preds = []
            for idx in range(sent_start_pos, eidx):
                ipt = seg_hiddens[idx, :].view(1, -1)
                ipt = torch.cat([intent_hidden, prev_state, ipt], dim=-1).view(1, 1, -1)
                ipt = self.dropout(ipt)
                if idx == 0:
                    out, (hx, cx) = self.lstm_layer.forward(ipt)
                else:
                    out, (hx, cx) = self.lstm_layer.forward(ipt, (prev_hx, prev_cx))
                prev_hx = hx
                prev_cx = cx
                pred = self.out_linear.forward(out)
                pred_prob = pred.softmax(-1)
                pred_idx = torch.argmax(pred_prob)
                # import logging
                # print(f"{pred_idx.shape=}")
                prev_state = self.slot_embedding_layer.forward(pred_idx).unsqueeze(0)
                # prev_state = pred
                # prev_cx = pred.view(1, -1) + cx
                # all_preds.append(pred_idx.detach().cpu().item())
                all_preds.append(pred.squeeze(0))
            # [L, num_slot]
            out = torch.cat(all_preds)
            return out

    @staticmethod
    def debug():
        model = SlotLSTMDecoder(num_slots=5, input_feats=800, num_intents=41)
        encoder_out = torch.randn(15, 800)
        intent_out = torch.argmax(torch.randn(41, ))
        model.forward(encoder_out, intent_out)


class Model(nn.Module):
    def __init__(self, args: Params, num_slots: int,
                 num_intents: int = 0, ):
        super(Model, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert)
        self.utter_encoder = nn.LSTM(
                bidirectional=True,
                input_size=BERT_DIM,
                hidden_size=args.utter_hidden
        )
        utter_out_dim = args.utter_hidden * 2
        self.history_encoder = nn.LSTM(
                bidirectional=True,
                input_size=utter_out_dim,
                hidden_size=args.hist_hidden,
                batch_first=True
        )
        histoty_out_dim = args.hist_hidden * 2
        self.pooling: Literal['max', 'mean'] = 'max'
        gcn_out_dim = args.gcn_out_dim
        lstm_out_dim = args.utter_hidden * 2
        self.gcn = GCN(in_size=histoty_out_dim, hid_size=args.gcn_hidden_dim,
                       out_size=gcn_out_dim, device=args.device, num_layers=args.gcn_layers)
        self.afl_bilinear = nn.Bilinear(in1_features=gcn_out_dim,
                                        in2_features=lstm_out_dim,
                                        out_features=1)
        self.intent_classifier = IntentClassifier(in_dim=gcn_out_dim + histoty_out_dim,
                                                  num_classes=num_intents)
        self.slot_decoder = SlotLSTMDecoder(num_slots=num_slots,
                                            input_feats=gcn_out_dim + utter_out_dim,
                                            num_intents=num_intents,
                                            dropout=args.dropout,
                                            slot_embed_dim=args.slot_embed_dim)

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    # @snoop
    def lstm_encoding(self, curr_turn, history=None):
        """Encoding curr turn and history at the same time
        :param curr_turn: [1, L, DIM]
        :param history: [T-1, L, DIM]
        :return: (utter_out, e_n) with shape (T, L, HDIM) and (1, T, HDIM)
        """
        # [T, L, DIM]

        if history is not None:
            # print(f"{history.shape=}")
            dialogues = torch.cat([history, curr_turn], dim=0)
        else:
            dialogues = curr_turn
        T, L, DIM = dialogues.size()
        utter_out, (_, _) = self.utter_encoder.forward(dialogues.view(L, T, DIM))
        # utter_out = utter_out.transpose(1, 0)

        if self.pooling == 'max':
            e_0, _ = torch.max(utter_out, dim=0, keepdim=True)
        else:
            e_0 = torch.mean(utter_out, dim=0, keepdim=True)
        # [1, T, DIM]
        # print(f"{e_0.shape=}")
        e_n, (_, _) = self.history_encoder.forward(e_0)
        return utter_out, e_n.squeeze(0)

    def ada_fusion_layer(self, history: torch.Tensor,
                         sentence: torch.Tensor):
        """

        :param history: [T, dim]
        :param sentence: [L, dim]
        :return: [L, dim+dim]
        """
        T = history.shape[0]
        L = sentence.shape[0]
        hist_ = history.unsqueeze(0).repeat(L, 1, 1)
        sent_ = sentence.unsqueeze(1).repeat(1, T, 1)
        a = self.afl_bilinear.forward(hist_, sent_)
        weights = a.squeeze(-1).softmax(-1)
        o = weights @ history
        res = torch.cat([sentence, o], dim=-1)
        return res

    # @snoop
    def graph_encoding(self, e_n: torch.Tensor):
        """

        :param e_n: [T, DIM]
        :return: [T, dim]
        """
        # device = e_n.device
        turn = e_n.size()[0]
        g = make_state_graph(turn, device=self.args.device)
        # print(f"{g.device=}")
        e_L = self.gcn.forward(g, e_n)
        return e_L

    @staticmethod
    @snoop
    def debug():
        args = Params().parse_args()
        model = Model(args, num_intents=41, num_slots=5)
        ct = torch.randn((1, 100, 768))
        hist = torch.randn((10, 100, 768))
        utter, e_n = model.lstm_encoding(ct, hist)
        # [T, dim]
        # e_L.shape = (11, 400)
        e_L = model.graph_encoding(e_n)
        last_sent_repr = utter[:, -1, :]
        # c.shape = (100, 1200)
        c = model.ada_fusion_layer(e_L, last_sent_repr)
        pred_intent = model.intent_classifier.forward(e_n, e_L)
        pred_intent = pred_intent.softmax(-1)
        intent_idx = torch.argmax(pred_intent)
        slot_probs = model.slot_decoder.forward(c, intent_idx)
        return pred_intent.item(), slot_probs

    # @snoop
    def forward(self, sentence_input: dict, history_ids: dict, intent_ids: torch.Tensor, eval_=False):
        """

        :param sentence_input:
        :param history_ids:
        :param intent_ids:
        :param eval_: if True, eval and return preds
        :return:
        """

        slot_labels, _ = sentence_input.pop("slot_ids"), history_ids.pop("slot_ids")
        sent_init_repr = self.bert.forward(**sentence_input)
        attn_mask = sentence_input['attention_mask']
        ct = sent_init_repr.last_hidden_state
        ct = ct.unsqueeze(0) if ct.dim() == 2 else ct
        if history_ids['input_ids'].nelement() != 0:
            history_ids = {k: v.squeeze(0) for k, v in history_ids.items()}
            hist_repr = self.bert.forward(**history_ids)
            hist = hist_repr.last_hidden_state
        else:
            hist = None
        utter, e_n = self.lstm_encoding(ct, hist)
        # [T, dim]
        # e_L.shape = (11, 400)
        e_L = self.graph_encoding(e_n)
        last_sent_repr = utter[:, -1, :]
        # c.shape = (100, 1200)
        c = self.ada_fusion_layer(e_L, last_sent_repr)
        pred_intent_org = self.intent_classifier.forward(e_n, e_L).unsqueeze(0)
        pred_intent = pred_intent_org.softmax(-1)
        intent_idx = torch.argmax(pred_intent)
        slot_probs = self.slot_decoder.forward(c, intent_idx)
        if attn_mask is not None:
            active_loss = attn_mask.view(-1) == 1
            active_logits = slot_probs[active_loss]
            active_labels = slot_labels.view(-1)[active_loss]
        else:
            active_logits = slot_probs
            active_labels = slot_labels
        intent_loss = tf.cross_entropy(pred_intent_org, intent_ids)
        slot_loss = tf.cross_entropy(active_logits, active_labels.squeeze(0))
        if eval_:
            return self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss, intent_idx, tf.softmax(active_logits, dim=-1)
        else:
            # return self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
            return intent_loss, slot_loss


if __name__ == '__main__':
    SlotLSTMDecoder.debug()
