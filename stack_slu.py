from collections import Counter
from typing import Literal,Optional

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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import math



from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_undirected #############

def device_watch(source, value):
    if hasattr(value, 'device'):
        return f"{source}.device", getattr(value, 'device')
    else:
        return f"{source} = ", "no device"


snoop.install(watch_extras=[device_watch])
BERT_DIM = 768


class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an embedding object.
        self.__embedding = EmbeddingCollection(
                self.__num_word,
                self.__args.word_embedding_dim
        )

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
                self.__args.word_embedding_dim,
                self.__args.encoder_hidden_dim,
                self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
                self.__args.word_embedding_dim,
                self.__args.attention_hidden_dim,
                self.__args.attention_output_dim,
                self.__args.dropout_rate
        )

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(
                self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                self.__args.intent_decoder_hidden_dim,
                self.__num_intent, self.__args.dropout_rate,
                embedding_dim=self.__args.intent_embedding_dim
        )
        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
                self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                self.__args.slot_decoder_hidden_dim,
                self.__num_slot, self.__args.dropout_rate,
                embedding_dim=self.__args.slot_embedding_dim,
                extra_dim=self.__num_intent
        )

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(
                self.__num_intent, self.__num_intent
        )
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:			    {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\tdimension of intent decoder hidden:        {};'.format(self.__args.intent_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_intent=None):
        # [N, L]
        word_tensor, _ = self.__embedding(text)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        # transformer_hiddens = self.__transformer(pos_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_intent = self.__intent_decoder(
                hiddens, seq_lens,
                forced_input=forced_intent
        )

        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        pred_slot = self.__slot_decoder(
                hiddens, seq_lens,
                forced_input=forced_slot,
                extra_input=feed_intent
        )

        if n_predicts is None:
            return tf.log_softmax(pred_slot, dim=1), tf.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cuda().data.numpy().tolist(), intent_index.cuda().data.numpy().tolist() #cuda

    def golden_intent_predict_slot(self, text, seq_lens, golden_intent, n_predicts=1):
        word_tensor, _ = self.__embedding(text)
        embed_intent = self.__intent_embedding(golden_intent)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_slot = self.__slot_decoder(
                hiddens, seq_lens, extra_input=embed_intent
        )
        _, slot_index = pred_slot.topk(n_predicts, dim=-1)

        # Just predict single slot value.
        return slot_index.cuda().data.numpy().tolist() #cuda


class EmbeddingCollection(nn.Module):
    """
    Provide word vector and position vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
                self.__input_dim, self.__embedding_dim
        )

        # Position vector encoder.
        # self.__position_layer = torch.zeros(self.__max_len, self.__embedding_dim)
        # position = torch.arange(0, self.__max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.__embedding_dim, 2) *
        #                      (-math.log(10000.0) / self.__embedding_dim))

        # Sine wave curve design.
        # self.__position_layer[:, 0::2] = torch.sin(position * div_term)
        # self.__position_layer[:, 1::2] = torch.cos(position * div_term)
        #
        # self.__position_layer = self.__position_layer.unsqueeze(0)
        # self.register_buffer('pe', self.__position_layer)

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Get position encoding.
        # position_x = Variable(self.pe[:, :input_x.size(1)], requires_grad=False)

        # Board-casting principle.
        return embedding_x, embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate, batch_first=True):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
                input_size=self.__embedding_dim,
                hidden_size=self.__hidden_dim,
                batch_first=batch_first,
                bidirectional=True,
                dropout=self.__dropout_rate,
                num_layers=1
        )

    def forward(self, embedded_text):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        # packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(dropout_text)
        # padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        # out = torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)
        return lstm_hiddens


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                    torch.randn(1, self.__embedding_dim),
                    requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=self.__hidden_dim,
                batch_first=True,
                bidirectional=False,
                dropout=self.__dropout_rate,
                num_layers=1
        )
        self.__linear_layer = nn.Linear(
                self.__hidden_dim,
                self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = tf.softmax(torch.matmul(
                linear_query,
                linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
                self.__input_dim, self.__input_dim, self.__input_dim,
                self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
                dropout_x, dropout_x, dropout_x
        )

        return attention_x


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

        layers = [dglnn.GraphConv(in_size, hid_size, activation=tf.relu).to(device)]
        # two-layer GCN
        for _ in range(1, num_layers - 1):
            layers.append(
                    dglnn.GraphConv(hid_size, hid_size, activation=tf.relu).to(device)
            )
        layers.append(dglnn.GraphConv(hid_size, out_size, activation=tf.relu).to(device))
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
        self.__num_intent = num_intents
        self.__args = args
        self.args = args
        #self.bert = BertModel.from_pretrained(args.bert)
        self.bert = BertModel.from_pretrained('/root/cgcn/cgcn_-af-master-gpu-gca/cgcn_-af-master/bert')
        self.utter_encoder = nn.LSTM(
                bidirectional=True,
                input_size=BERT_DIM,
                hidden_size=args.utter_hidden
        )
        utter_out_dim = args.utter_hidden * 2
        self.history_encoder = nn.LSTM(
                bidirectional=True,
                input_size=self.args.encoder_hidden_dim,
                hidden_size=args.hist_hidden,
                batch_first=True
        )
        histoty_out_dim = args.hist_hidden * 2
        self.pooling: Literal['max', 'mean'] = 'max'
        gcn_out_dim = args.gcn_out_dim
        lstm_out_dim = args.utter_hidden * 2
        self.gcn = GCN(in_size=histoty_out_dim, hid_size=args.gcn_hidden_dim,
                       out_size=histoty_out_dim, device=0, num_layers=args.gcn_layers)  #args.device
        self.afl_bilinear = nn.Bilinear(in1_features=histoty_out_dim,
                                        in2_features=self.args.encoder_hidden_dim,
                                        out_features=1)
        self.intent_classifier = IntentClassifier(in_dim=histoty_out_dim + histoty_out_dim,
                                                  num_classes=num_intents)
        self.slot_decoder = SlotLSTMDecoder(num_slots=num_slots,
                                            input_feats=histoty_out_dim + self.args.encoder_hidden_dim,
                                            num_intents=num_intents,
                                            dropout=args.dropout,
                                            slot_embed_dim=args.slot_embed_dim)
        # MARK: check intent
        self.intent_embedding = nn.Embedding(num_embeddings=num_intents,
                                             embedding_dim=num_intents)
        self.intent_embedding.weight.data = torch.eye(num_intents)
        self.intent_embedding.weight.requires_grad = False
        self.slot_mlp = nn.Linear(in_features=gcn_out_dim + utter_out_dim + num_intents,
                                  out_features=num_slots,
                                  bias=False)
        # MARK: stack slu
        # Initialize an embedding object.

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
                BERT_DIM,
                self.args.encoder_hidden_dim,
                self.args.dropout
        )

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(
                self.args.encoder_hidden_dim + histoty_out_dim,
                self.args.intent_decoder_hidden_dim,
                self.__num_intent, self.__args.dropout,
                embedding_dim=self.args.num_intent_embed
        )
        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
                self.args.encoder_hidden_dim + histoty_out_dim + BERT_DIM,
                self.args.slot_decoder_hidden_dim,
                num_slots, self.args.dropout,
                embedding_dim=self.__args.slot_embed_dim,
                extra_dim=self.__num_intent
        )

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(
                self.__num_intent, self.__num_intent
        )
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()


        #GCA
        self.encoder1 =GEncoder(in_channels=256, out_channels=256, activation=torch.nn.PReLU(),base_model=GCNConv, k=2,skip=False)# 编码 dataset.num_features=x的特征数 （个数，每个事件的特征数） #调用问题？num_feastures=opt.d_k=64 此处的dk应为nf
        self.grace = GRACE(encoder=self.encoder1,num_hidden=256, num_proj_hidden=128, tau= 0.5)  #num_proj_hidden不用动

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    @property
    def other_params(self):
        result = []
        for name, param in self.named_parameters(recurse=True):
            if not name.startswith('bert'):
                result.append(param)
        return result

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
        utter_out = self.__encoder.forward(dialogues)
        # utter_out = utter_out.transpose(1, 0)

        if self.pooling == 'max':
            e_0, _ = torch.max(utter_out, dim=1, keepdim=True)
        else:
            e_0 = torch.mean(utter_out, dim=1, keepdim=True)
        # [1, T, DIM]
        # print(f"{e_0.shape=}")
        e_n, (_, _) = self.history_encoder.forward(e_0)
        return utter_out, e_n.squeeze(1)

    def ada_fusion_layer(self, history: torch.Tensor,
                         sentence: torch.Tensor):
        """

        :param history: [T, dim]
        :param sentence: [L, dim]
        :return: [L, dim+dim]
        """
        T = history.shape[0]
        L = sentence.shape[0]
        # print(f"{history=}")
        hist_ = history.unsqueeze(0).repeat(L, 1, 1)
        sent_ = sentence.unsqueeze(1).repeat(1, T, 1)
        a = self.afl_bilinear.forward(hist_, sent_)
        a = tf.leaky_relu(a)
        weights = a.squeeze(-1).softmax(-1)
        # print(f"{weights=}")
        o = torch.mm(weights, history)
        # print(f"{o=}")
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
        g = make_state_graph(turn, device=0)####self.args.device
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
    def stack_forward(self, sentence_input: dict, history_ids: dict,
                      intent_ids: torch.Tensor, eval_=False,
                      forced_slot=None, forced_intent=None):
        slot_labels, _ = sentence_input.pop("slot_ids"), history_ids.pop("slot_ids")
        sent_init_repr = self.bert.forward(**sentence_input)
        attn_mask = sentence_input['attention_mask']

        # [L, DIM]
        sent_embeds = sent_init_repr.last_hidden_state

        if history_ids['input_ids'].nelement() != 0: #fou   #在错误数据的时候这里是是 明天看看是不是这个位置的原因
            history_ids = {k: v.squeeze(0) for k, v in history_ids.items()}
            hist_repr = self.bert.forward(**history_ids)
            hist = hist_repr.last_hidden_state
        else:
            hist = None
        utter, e_n = self.lstm_encoding(sent_embeds, hist)
        e_L = self.graph_encoding(e_n)  #IndexError: index 1 is out of bounds for dimension 0 with size 1
        """if len(e_L.shape)==2:
            EE=e_L.unsqueeze(2)
            e_L=EE.repeat(1, 1,2)
            
            
            e_L=torch.cat((e_L,e_L),0)
            print(e_L.shape)"""
        el2=torch.rand(1, 512)
        for i in range(512):
            if e_L[0][i]==0:
                el2[0][i]=0
            else:
                el2[0][i]=1
        e_Lint=torch.as_tensor(el2, dtype=torch.int64).cuda()
        if e_L.shape[0]==1:

            
            e_L=torch.cat((e_L,e_Lint),0).cuda()
            #print(e_L)
        else:
            #e_L=torch.cat((e_L[0],e_Lint),0)
            e_L[1]=e_Lint
        last_sent_repr = utter[-1, :, :]



        for i in range(3):
            self.gca(last_sent_repr,e_L)    

        last_sent_repr_gca=self.test(last_sent_repr,e_L)

        e_L = self.graph_encoding(e_n)

        c = self.ada_fusion_layer(e_L, last_sent_repr_gca)
        # lstm_hiddens = self.__encoder.forward(sent_embeds)
        #print(c.shape)
        #print(sent_embeds.shape)
        # hiddens = torch.cat([sent_embeds, utter[-1, :, :].unsqueeze(0)], dim=-1).squeeze(0)
        hiddens = torch.cat([c, sent_embeds.squeeze(0)], dim=-1)
        seq_lens = (attn_mask == 1).sum()
        pred_intent = self.__intent_decoder.forward(
                c, forced_input=forced_intent,
                seq_lens=[seq_lens.item()]
        )
        if self.args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        pred_slot = self.__slot_decoder(
                hiddens[:seq_lens.item(), :], [seq_lens.item()],
                forced_input=forced_slot,
                extra_input=feed_intent
        )
        active_loss = attn_mask.view(-1) == 1
        active_logits = pred_slot
        active_labels = slot_labels
        intent_loss = self.criterion(pred_intent, intent_ids.repeat(seq_lens.item()))
        slot_loss = self.criterion(active_logits, active_labels.squeeze(0)[active_loss])
        intent_idx = torch.argmax(tf.softmax(pred_intent, dim=-1), dim=-1)
        pred_intent__ = Counter(intent_idx.tolist()).most_common()[0][0]
        # print(f"{intent_idx=}")
        if eval_:
            return intent_loss, slot_loss, torch.tensor(pred_intent__, dtype=torch.long), tf.softmax(active_logits, dim=-1)
        else:
            # return self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
            return intent_loss, slot_loss

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    

    def gca(self,x, edge):  # num_feastures,optimizer,

        self.grace.train()
        #optimizer.zero_grad()

        def drop_feature(x, drop_prob):
            drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
            x = x.clone()
            x[:, drop_mask] = 0

            return x


        def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
            w = w / w.mean() * p
            w = w.where(w < threshold, torch.ones_like(w) * threshold)
            drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

            drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

            x = x.clone()
            x[drop_mask] = 0.

            return x

        def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
            edge_weights = edge_weights / edge_weights.mean() * p
            edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
            sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

            return edge_index[:, sel_mask]

        def degree_drop_weights(edge_index):
            edge_index_ = to_undirected(edge_index)
            deg = degree(edge_index_[1])  #RuntimeError: scatter(): Expected dtype int64 for index
            deg_col = deg[edge_index[1]].to(torch.float32)
            s_col = torch.log(deg_col)
            weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

            return weights

        def drop_edge(idx: int):
            # global drop_weights
            if idx == 1:
                return drop_edge_weighted(edge, drop_weights, p=0.3, threshold=0.7)#.cuda()
            else:
                return drop_edge_weighted(edge, drop_weights, p=0.4, threshold=0.7)#.cuda()

        el=torch.rand(2, 512)
        for i in range(512):
            if edge[0][i]==0:
                el[0][i]=el[1][i]=0
            else:
                el[0][i]=el[1][i]=int(edge[0][i]*100)
        eLint=torch.as_tensor(el, dtype=torch.int64)

        drop_weights = degree_drop_weights(eLint)
        edge_index_ = to_undirected(edge)
        node_deg = degree(eLint[0])


        edge_index_1 = drop_edge(1)  # 删除了几个edge连接  改为整形即可
        x_long=x.shape[0]-1
        for i in range(0,edge_index_1.shape[0]):
            for j in range(edge_index_1.shape[1]):
                if int(edge_index_1[i][j]*100)<=x_long:
                    edge_index_1[i][j]=int(edge_index_1[i][j]*100)
                else:
                    edge_index_1[i][j]=x_long
        edge_index_1=torch.as_tensor(edge_index_1, dtype=torch.int64)
        edge_index_2 = drop_edge(2)
        for i in range(0,edge_index_2.shape[0]):
            for j in range(edge_index_2.shape[1]):
                if int(edge_index_2[i][j]*100)<=x_long:
                    edge_index_2[i][j]=int(edge_index_2[i][j]*100)
                else:
                    edge_index_2[i][j]=x_long
        edge_index_2=torch.as_tensor(edge_index_2, dtype=torch.int64)
        x_1 = drop_feature(x, 0.1)
        x_2 = drop_feature(x, 0.0)
        #print(x_1.shape)
        #print(edge_index_1.shape)
        z1 = self.grace(x_1, edge_index_1)
        z2 = self.grace(x_2, edge_index_2)

        loss1 = self.grace.loss(z1, z2, batch_size=None)
        
        loss1.backward(retain_graph=True)
        # else:
        #     loss1.backward()
        #optimizer.step()

    def test(self, x, edge):
        self.grace.eval()
        x_long=x.shape[0]-1
        for i in range(0,edge.shape[0]):
            for j in range(edge.shape[1]):
                if int(edge[i][j]*100)<=x_long:
                    edge[i][j]=int(edge[i][j]*100)
                else:
                    edge[i][j]=x_long
        edge=torch.as_tensor(edge, dtype=torch.int64)
        z = self.grace(x, edge)#.cuda()
        return z



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
        last_sent_repr = utter[-1, :, :]




        #修改     last_sent_repr为节点 e_l为边 ######################################
        #if 
        for i in range(3):
            self.gca(last_sent_repr,e_L)    

        last_sent_repr_gca=self.test(last_sent_repr,e_L)





        # c.shape = (100, 1200)
        c = self.ada_fusion_layer(e_L, last_sent_repr_gca)
        pred_intent_org = self.intent_classifier.forward(e_n, e_L).unsqueeze(0)
        pred_intent = pred_intent_org.softmax(-1)
        intent_idx = torch.argmax(pred_intent)
        # MARK: mod here
        if self.args.slot_mlp:
            intent_embed = self.intent_embedding.forward(intent_idx).view(1, -1)
            L = c.shape[0]
            slot_inpt = torch.cat([c, intent_embed.repeat(L, 1)], dim=-1)
            slot_probs = self.slot_mlp.forward(slot_inpt)
        else:
            slot_probs = self.slot_decoder.forward(c, intent_idx)
        if attn_mask is not None:
            active_loss = attn_mask.view(-1) == 1
            active_logits = slot_probs[active_loss]
            active_labels = slot_labels.view(-1)[active_loss]
        else:
            active_logits = slot_probs
            active_labels = slot_labels
        # print(f"{active_labels=}")
        intent_loss = self.criterion(pred_intent_org, intent_ids)
        slot_loss = self.criterion(active_logits, active_labels.squeeze(0))
        if eval_:
            return intent_loss, slot_loss, intent_idx, tf.softmax(active_logits, dim=-1)
        else:
            # return self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
            return intent_loss, slot_loss

class GEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(GEncoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                #print(x.shape,edge_index.shape)
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    def __init__(self, encoder: GEncoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: GEncoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = tf.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = tf.normalize(z1)
        z2 = tf.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret





if __name__ == '__main__':
    SlotLSTMDecoder.debug()
