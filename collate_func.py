from typing import List


import numpy as np

import torch


def collate_to_max_length(batch: List[List[torch.Tensor]],mode, max_len_target: int=None,max_len_prior: int =None,fill_values: List[float]=None) -> List[torch.Tensor]:

    max_bert_hidden_size=768
    max_len_target=max_len_prior=40
    
    data_field=['target_idx','claim_set_idx_list','target_claim',
                'prior_idx','prior_passage_indx_list','prior_passage',
                'target_represent','prior_represent','label']

    # lengths=np.array([[len(sample['target']),len(sample['prior']),len(sample['labels'])]for sample in batch])
    lengths = np.array([[len(sample[idx]) for idx in data_field[-3:]] for sample in batch])

    # target_claim,prior_claim,label=list(zip(*batch)) 
    
    batch_size, num_fields= lengths.shape
    # fill_values = fill_values or [0.0] * num_fields

    max_lengths = lengths.max(axis=0)

    if max_len_target:
        max_lengths[0]=max_len_target
    if max_len_prior:
        max_lengths[1]=max_len_prior
    output=[]
    if mode=='test':
        output.extend([sample[idx] for sample in batch] for idx in data_field[:-3])

    output.append(torch.full([batch_size,max_lengths[0]+max_lengths[1],max_bert_hidden_size],
                       fill_value=0.0,
                       dtype=batch[0]['target_represent'].dtype))
    output.append(torch.full([batch_size,max_lengths[2]],
                             fill_value=0.0,
                             dtype=batch[0]['label'].dtype))
    # if max_len_target:
    #     # assert max_lengths.max() <= max_len_target
    #     max_lengths = np.ones_like(max_lengths) * max_len_target

    # output = torch.full([batch_size, max_lengths[0]],
    #                      fill_value=fill_values[0],
    #                      dtype=batch[0][0].dtype)

    # output = [torch.full([batch_size, max_lengths[field_idx]],
    #                      fill_value=fill_values[field_idx],
    #                      dtype=batch[0][field_idx].dtype)
    #           for field_idx in range(num_fields-1)]
    # output=torch.full([batch_size, max_lengths[0]],
    #                   fill_value=fill_values[0],
    #                   dtype=batch[0]['target'].dtype)
    
    
    # return output
    # print(output)
    for sample_idx in range(batch_size):
        # for field_idx in range(num_fields):

    #         # seq_length
        data_target = batch[sample_idx]['target_represent']
        data_prior  = batch[sample_idx]['prior_represent']
        data_labels = batch[sample_idx]['label']
        output[-2][sample_idx][: data_target.shape[0]] = data_target
        output[-2][sample_idx][max_lengths[0]: max_lengths[0]+data_prior.shape[0]] = data_prior
        output[-1][sample_idx][:data_labels.shape[0]]=data_labels

    # return output
    # # generate span_index and span_mask
    # max_sentence_length = max_lengths[0]
    start_indexs = []
    end_indexs = []
    for i in range(max_lengths[0]):
        for j in range(max_lengths[0],max_lengths[1]+max_lengths[0]):
            # # span大小为10
            # if j - i > 10:
            #     continue
            start_indexs.append(i)
            end_indexs.append(j)
    # # generate span mask
    span_masks = []
    for sample in batch:
        span_mask = []
        # # middle_index = max_lengths
        # len_index=max_lengths[0]+sample['prior_represent'].shape[0]
        # for start_index, end_index in zip(start_indexs, end_indexs):
        #     if start_index < len_index and end_index < len_index and (
        #         start_index >= max_lengths[0] or end_index < sample['target_represent'].shape[0]):
        #         span_mask.append(0)
        #     else:
        #         span_mask.append(1e6)
        # span_masks.append(span_mask)
        len_index=max_lengths[0]+sample['prior_represent'].shape[0]
        for start_index, end_index in zip(start_indexs, end_indexs):
            if start_index < sample['target_represent'].shape[0] and end_index < len_index:
                span_mask.append(0)
            else:
                span_mask.append(1e6)
        span_masks.append(span_mask)

    # # add to output
    output.append(torch.LongTensor(start_indexs))
    output.append(torch.LongTensor(end_indexs))
    output.append(torch.LongTensor(span_masks))
    return output  # target_idx, prior_idx, claim_set_idx_list, prior_passage_indx_list, input, label, start_indexs, end_indexs, span_masks





