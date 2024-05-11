import torch
import numpy as np
from demo.collate_func import collate_to_max_length
from demo.dataset import Patent_Dataset
from functools import partial

# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
from torch.utils.data.dataloader import DataLoader
import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
# from transformers import AutoTokenizer,AutoModel

# from datasets.collate_functions import collate_to_max_length
from demo.model import ExplainableModel
from utils.radom_seed import set_random_seed
from torchmetrics import Accuracy



class ExplainNLP(pl.LightningModule):

    def __init__(
        self,
        config : dict
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.config = config
        # self.args = args
        # if isinstance(args, argparse.Namespace):
        #     self.save_hyperparameters(args)
        # self.bert_dir = args.bert_path
        self.model = ExplainableModel(self.config)
        # self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_dir)
        # self.tokenizer=AutoTokenizer.from_pretrained('roberta-base')
        # self.pretrained_model=AutoModel.from_pretrained('distilroberta-base', return_dict = True)

        self.loss_fn = CrossEntropyLoss()
        # self.train_acc = pl.metrics.Accuracy()
        # self.valid_acc = pl.metrics.Accuracy()
        self.train_acc = Accuracy(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.output = []
        self.check_data = []

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          betas=(0.9, 0.98),  # according to RoBERTa paper
                          lr=self.config['lr'],
                          eps=self.config['adam_epsilon'])
        # t_total = len(self.train_dataloader()) // self.config['accumulate_grad_batches'] * self.config['max_epochs']
        t_total = len(self.train_dataloader()) // self.config['batch_size'] * self.config['max_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config['warmup_steps'],
                                                    num_training_steps=t_total)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_rep, start_indexs, end_indexs, span_masks):
        return self.model(input_rep, start_indexs, end_indexs, span_masks)

    def compute_loss_and_acc(self, batch, mode='train'):

        if mode == 'test':
            target_idx,claim_set_idx_list,target_claim,prior_idx,prior_passage_indx_list,prior_passage, input, labels, start_indexs, end_indexs, span_masks = batch
        else:
            input, labels, start_indexs, end_indexs, span_masks = batch
        y = labels.view(-1)
        y_hat, a_ij = self.forward(input, start_indexs, end_indexs, span_masks)
        # compute loss
        ce_loss = self.loss_fn(y_hat, y)
        reg_loss = self.config['lamb'] * a_ij.pow(2).sum(dim=1).mean()
        loss = ce_loss - reg_loss
        # compute acc
        predict_scores = F.softmax(y_hat, dim=1)
        predict_labels = torch.argmax(predict_scores, dim=-1)
        # predict_labels = torch.LongTensor([predict_scores>0.5])
        if mode == 'train':
            acc = self.train_acc(predict_labels, y)
        else:
            acc = self.valid_acc(predict_labels, y)
        # if test, save extract spans
        if mode == 'test':
            values, indices = torch.topk(a_ij, self.config['span_topk'])
            values = values.tolist()
            indices = indices.tolist()
            for i in range(len(values)):
                # input_list = input[i].tolist()

                # origin_sentence = self.tokenizer.decode(input_ids_list, skip_special_tokens=True)

                self.output.append(
                    str(labels[i].item()) + '<->' + str(predict_labels[i].item()) + '<->'+target_idx[i]+'&'+ prior_idx[i] + '\n')
                # print()
                for j, span_idx in enumerate(indices[i]):
                    score = values[i][j]
                    start_index = start_indexs[span_idx]
                    end_index = end_indexs[span_idx]
                    # pre = self.tokenizer.decode(input_ids_list[:start_index], skip_special_tokens=True)
                    # high_light = self.tokenizer.decode(input_ids_list[start_index:end_index + 1],
                    #                                    skip_special_tokens=True)
                    # post = self.tokenizer.decode(input_ids_list[end_index + 1:], skip_special_tokens=True)
                    # span_sentence = pre + '【' + high_light + '】' + post
                    prior_passage_indx=prior_passage_indx_list[i][end_index%40] # start_passage_num  is 1

                    self.output.append(format('%.4f' % score) + "->" + ('Claim'+", ".join(claim_set_idx_list[i][start_index]))+
                                       ' & Related passage: ' +prior_passage_indx[0]+'-'+str(prior_passage_indx[1]) + '\n')
                    self.output.append('Target_text: '+" ".join([target_claim[i][key] for key in claim_set_idx_list[i][start_index]])+'\n')
                    self.output.append('Prior_text: '+ prior_passage[i][prior_passage_indx[0]][prior_passage_indx[1]-1]+'\n')
                    # # print(format('%.4f' % score), "->", span_sentence)
                    if j == 0:
                    #     # generate data for check progress
                        self.check_data.append(str(labels[i].item()) + '\t' + str(start_index)+'&' +str(end_index%40) + '\n')
                self.output.append('\n')
            # print('='*30)

        return loss, acc

    # def validation_epoch_end(self, outs):
    #     # log epoch metric
    #     self.valid_acc.compute()
    #     self.log('valid_acc_end', self.valid_acc.compute())
    def on_validation_epoch_end(self):
        # log epoch metric
        self.valid_acc.compute()
        self.log('valid_acc_end', self.valid_acc.compute())

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def training_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_acc(batch, mode='dev')
        self.log('valid_acc', acc, on_step=False, on_epoch=True)
        self.log('valid_loss', loss)
        return loss

    def get_dataloader(self, prefix="train") -> DataLoader:
        """get training dataloader"""
        # if self.args.task == 'snli':
        #     dataset = SNLIDataset(directory=self.args.data_dir, prefix=prefix,
        #                           bert_path=self.bert_dir,
        #                           max_length=self.args.max_length)
        # else:
        #     dataset = SSTDataset(directory=self.args.data_dir, prefix=prefix,
        #                          bert_path=self.bert_dir,
        #                          max_length=self.args.max_length)
        # dataset=Patent_Dataset('demo\demo_basic_dataset.txt')
        # dataloader = DataLoader(
        #     dataset=dataset,
        #     batch_size=10,
        #     num_workers=0,

        # )
        if prefix=="train":
          patent_ds = Patent_Dataset('demo/train_B60only.txt',mode=prefix)
        else:
          patent_ds = Patent_Dataset('demo/test_B60only.txt',mode=prefix)
        dataloader = DataLoader(
                    dataset=patent_ds,
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['workers'],
                    collate_fn=partial(collate_to_max_length,mode=prefix),
                    drop_last=False
                )
        return dataloader

    def test_dataloader(self):
      return self.get_dataloader("test")

    def test_step(self, batch, batch_idx):
      loss, acc = self.compute_loss_and_acc(batch, mode='test')
      return {'test_loss': loss, "test_acc": acc}

    def on_test_epoch_end(self):
        with open(os.path.join(self.config['save_path'], 'output.txt'), 'w', encoding='utf8') as f:
            f.writelines(self.output)
        with open(os.path.join(self.config['save_path'], 'test.txt'), 'w', encoding='utf8') as f:
            f.writelines(self.check_data)
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        # tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        # print(avg_loss, avg_acc)
        # return {'val_loss': avg_loss, 'log': tensorboard_logs}
        return {'val_loss': 0, 'log': 0}
    
config = {
    'model_name': 'demo_base',
    'lr':2e-5,
    'batch_size':5,
    'workers':0,
    'weight_decay':0.0,
    'adam_epsilon':1e-9,
    'warmup_steps':0,
    'use_memory':'store_true',
    'max_length':512,
    'checkpoint_path':'lightning_logs/version_3/checkpoints/epoch=0-step=80.ckpt',
    # 'data_dir':'demo/demo_basic_dataset.txt',
    'save_path':'save_demo',
    'save_topk':5,
    'span_topk':5,
    'lamb':1.0,
    'mode':'train',
    'max_epochs':1
}


def train(config):
    # if save path does not exits, create it
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])

    model = ExplainNLP(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath =os.path.join(config['save_path'], '{epoch}-{valid_loss:.4f}-{valid_acc_end:.4f}'),
        save_top_k=config['save_topk'],
        save_last=True,
        monitor="valid_acc_end",
        mode="max",
    )
    # logger = TensorBoardLogger(
    #     save_dir=config['save_path'],
    #     name='log'
    # )

    # save args
    # with open(os.path.join(args.save_path, "args.json"), 'w') as f:
    #     args_dict = args.__dict__
    #     del args_dict['tpu_cores']
    #     json.dump(args_dict, f, indent=4)

    # trainer = Trainer.from_argparse_args(config,
    #                                      checkpoint_callback=checkpoint_callback,
    #                                      distributed_backend="ddp",
    #                                      logger=logger)
    trainer = Trainer(max_epochs=config['max_epochs'],num_sanity_val_steps=1)
    trainer.fit(model)

def evaluate(config):
    model = ExplainNLP(config)
    checkpoint = torch.load(config['checkpoint_path'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer(max_epochs=config['max_epochs'])
    trainer.test(model)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train(config)
















