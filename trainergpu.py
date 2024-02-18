import os.path

import numpy
import torch
from stack_slu import Model
from params import Params
from dataloader import Dataset
from torch.utils.data import DataLoader as TorchDataloader
from torch.optim import Adam
from utils import get_logger, get_sentence_frame_acc, get_slot_metrics
import time
from tqdm import tqdm, trange
from torchmetrics.classification import MulticlassAccuracy
from transformers.optimization import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, args: Params):
        self.args = args
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if not os.path.exists('log'):
            os.mkdir('log')
        self.logger = get_logger(os.path.join('log', f"{args.dataset}_{now}.log"))
        self.args.seed_anything()
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warn = self.logger.warn
        self.info("Loading datasets...")
        self.train_dataset = Dataset(args, status='train')
        self.test_dataset = Dataset(args, status='test')
        self.valid_dataset = Dataset(args, status='dev')
        self.info("Define model...")
        if self.args.stack:
            self.info("Testing stack SLU...")
            # from model import Model
        self.model = Model(args, num_slots=self.train_dataset.num_slots,
                           num_intents=self.train_dataset.num_intents)
        self.model.to(args.device)
        self.optim = Adam(params=self.model.other_params, lr=self.args.fast_lr,
                          weight_decay=1e-6)
        self.bert_optim = Adam(params=self.model.bert.parameters(),
                               lr=self.args.slow_lr,
                               weight_decay=1e-6)
        self.warmup_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.bert_optim,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=int(len(self.train_dataset) * args.max_epoch)
        )
        self.warmup_scheduler_2 = get_linear_schedule_with_warmup(
                optimizer=self.optim,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=int(len(self.train_dataset) * args.max_epoch)
        )
        self.micro_acc = MulticlassAccuracy(num_classes=self.train_dataset.num_intents, average='micro')
        self.macro_acc = MulticlassAccuracy(num_classes=self.train_dataset.num_intents, average='macro')
        self.freezed = False
        if os.path.exists(os.path.join(self.args.save_dir, self.args.ckpt)): #false
            self.critical(f"Loading from {os.path.join(self.args.save_dir, self.args.ckpt)}")
            res = torch.load(os.path.join(self.args.save_dir, self.args.ckpt),
                             map_location=self.args.device)
            self.model.load_state_dict(res['model'])
        self.info(f"{self.model=}")

    def fit(self):
        self.info(f"Start fitting with {self.args.as_dict()}...")

        accum_iter = self.args.accumulated_steps

        epbar = trange(self.args.max_epoch)
        epoch_loss = 0.0
        total_steps = 0
        backward_cnt = 0
        for epoch in epbar:
            if not self.model.training:
                self.model.train()
            batch_loss = []
            train_dataloader = TorchDataloader(self.train_dataset,
                                               shuffle=False, batch_size=1)   #######shuffle=True
            pbar = tqdm(train_dataloader, desc="Train")
            for batch_idx, (sentence_input, history_ids, intent_ids) in enumerate(pbar):
                # forward pass
                if self.args.stack:
                    intent_loss, slot_loss = self.model.stack_forward(sentence_input, history_ids, intent_ids)
                else:
                    intent_loss, slot_loss = self.model.forward(sentence_input, history_ids, intent_ids)
                loss = self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
                # loss = loss / accum_iter
                pbar.set_postfix({
                    'sloss': f"{slot_loss.detach().item():.4f}",
                    'iloss': f"{intent_loss.detach().item():.4f}"
                })
                loss.backward()  #明天再跑一遍，在这一步之前看一眼  "curr_turn":为0到10 等于1时报错
                # backward_cnt += 1
                # batch_loss.append(loss.detach().item())
                epoch_loss =epoch_loss +loss.detach().item()
                # backward pass
                # weights update
                #loss.backward()
                self.optim.step()
                self.bert_optim.step()
                self.warmup_scheduler.step()
                self.warmup_scheduler_2.step()
                self.model.zero_grad()
                total_steps = total_steps+1

                if total_steps % self.args.eval_steps == 0 and total_steps != 0:
                    self.eval(curr_step=total_steps, test=True)
                if total_steps % self.args.save_steps == 0 and total_steps != 0:
                    self.save()
                if not self.freezed and total_steps > self.args.fine_tune_steps:
                    self.model.freeze_bert()
                    self.freezed = True

            epbar.set_postfix({'eloss': f"{epoch_loss:.6f}"})

    def save(self):
        result = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'args':  self.args.as_dict()
        }
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        torch.save(result, os.path.join(self.args.save_dir, f"model_{now}.ckpt"))

    def eval(self, test=False, samples: int = -1, curr_step: int = 0):
        dataset__ = self.test_dataset if test else self.valid_dataset
        dlr = TorchDataloader(dataset__, batch_size=1, shuffle=False)
        pbar = tqdm(dlr, desc="Test" if test else "Valid")
        self.model.eval()
        slot_preds = []
        slot_pred_syms = []
        slot_targets = []
        slot_target_syms = []
        intent_preds = []
        intent_target = []
        decoded_text = []
        eval_loss = 0.0





        torch.backends.cudnn.enabled = False  ############解决 RuntimeError: cudnn RNN backward can only be called in training mode




        for batch_idx, (sentence_input, history_ids, intent_ids) in enumerate(pbar):
            attn_mask = sentence_input['attention_mask']
            # sent_ids = sentence_input["input_ids"]
            # sent_sym = self.train_dataset.tokenizer.convert_ids_to_tokens(sent_ids.squeeze())
            # print(sent_sym)
            if 0 < samples < batch_idx:
                break
            slot_ids = sentence_input["slot_ids"]
            # self.critical(f"{slot_ids.cpu().tolist()=}")
            # self.critical(f"{attn_mask.cpu().tolist()=}")
            if self.args.stack:
                intent_loss, slot_loss, intent_pred, active_logits = self.model.stack_forward(sentence_input,
                                                                                              history_ids,
                                                                                              intent_ids,
                                                                                              eval_=True)
                pbar.set_postfix({
                    'sloss': f"{slot_loss.detach().item():.6f}",
                    'iloss': f"{intent_loss.detach().item():.6f}"
                })
                loss = self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
            else:
                intent_loss, slot_loss, intent_pred, active_logits = self.model.forward(sentence_input,
                                                                                        history_ids,
                                                                                        intent_ids,
                                                                                        eval_=True)
                pbar.set_postfix({
                    'sloss': f"{slot_loss.detach().item():.6f}",
                    'iloss': f"{intent_loss.detach().item():.6f}"
                })
                loss = self.args.alpha_1 * intent_loss + self.args.alpha_2 * slot_loss
            eval_loss += loss.detach().cuda().item()   #cuda
            active_loss = attn_mask.view(-1) == 1
            # active_logits = slot_prob[active_loss]
            active_slots = slot_ids.squeeze()[active_loss]
            # self.critical(f"{active_logits=}")
            slot_pred = torch.argmax(active_logits,
                                     dim=-1)
            slot_preds.append(slot_pred.detach().cuda().tolist()) #cuda
            slot_pred_syms.append(list(map(self.train_dataset.ids2slot.get,
                                           slot_pred.detach().cuda().tolist())))#cuda
            slot_target_syms.append(list(map(self.train_dataset.ids2slot.get,
                                             active_slots.detach().cuda().tolist())))#cuda

            slot_targets.append(active_slots.cpu().detach().tolist())#cpu
            intent_preds.append(intent_pred.detach())
            intent_target.append(intent_ids.detach())
        micro_acc = self.micro_acc(torch.stack(intent_preds).cpu(),#cpu
                                   torch.cat(intent_target).cpu())
        macro_acc = self.macro_acc(torch.stack(intent_preds).cpu(),
                                   torch.cat(intent_target).cpu())
        semantic_acc = get_sentence_frame_acc(
                intent_preds=torch.stack(intent_preds).cpu().numpy(),
                intent_labels=torch.cat(intent_target).cpu().numpy(),
                slot_labels=slot_targets,
                slot_preds=slot_preds
        )
        slot_metrics = get_slot_metrics(slot_pred_syms, slot_target_syms)
        # write slots
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        with open(f'outputs/{self.args.dataset}_slot_{curr_step}.txt', 'w') as fp:
            for pred, target in zip(slot_pred_syms, slot_target_syms):
                fp.write(f"[P] {' '.join(pred)}\n")
                fp.write(f"[T] {' '.join(target)}\n")
                fp.write('----------------------\n')
        with open(f"outputs/{self.args.dataset}_intent_{curr_step}.txt", 'w') as fp:
            for pred, label in zip(torch.stack(intent_preds).cuda().tolist(), #cuda
                                   torch.cat(intent_target).cuda().tolist()):
                fp.write(f'[P] {self.train_dataset.ids2intent[pred]}\t'
                         f'[T] {self.train_dataset.ids2intent[label]}\n')

        if test:
            self.critical("Result on TEST dataset:")
        else:
            self.critical("Result on Valid dataset:")
        self.critical(f"{macro_acc=}")
        self.critical(f"{micro_acc=}")
        self.critical(f"{semantic_acc=}")
        for k, v in slot_metrics.items():
            self.critical(f"{k}: {v}")
        self.model.train()


if __name__ == '__main__':
    cfg = Params().parse_args()
    trainer = Trainer(args=cfg)
    trainer.eval(samples=10)
    trainer.fit()
