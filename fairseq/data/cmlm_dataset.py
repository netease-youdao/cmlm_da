
import numpy as np
import random
import torch

from . import data_utils, FairseqDataset


CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'

def random_mask(tokens, prob=0.15):
    for i, tok in enumerate(tokens):
        if random.random() < prob:
            tokens[i] = MASK

    return tokens

def convert_token_to_bert(token, convert=False):
    if not convert:
        return token
    raise NotImplementedError

def random_word(tokens, output_vocab, mask_prob=0.15):
    """
    NOTE: this assumes other MT prepro like moses and we try to align
          them with BERT
    Masking some random tokens for Language Model task with probabilities as in
    the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param output_vocab: vocab for seq2seq output
    :return: (list of str, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        # mask token with 15% probability
        if random.random() < mask_prob:
            # we always MASK given our purpose
            tokens[i] = MASK

            # append current token to output (we will predict these later)
            try:
                output_label.append(output_vocab.index(token))
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(output_vocab.unk)
        else:
            # handle input for BERT
            tokens[i] = convert_token_to_bert(token)

            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    # last SEP is used to learn EOS
    if random.random() < mask_prob:
      tokens.append(MASK)
      output_label.append(output_vocab.eos())
    else:
      tokens.append(SEP)
      output_label.append(-1)

    return tokens, output_label

def collate(
        samples, pad_idx, eos_idx,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])

    input_ids = merge('input_ids', left_pad=False)
    ntokens = sum(len(s['input_ids']) for s in samples)

    input_mask = merge('input_mask', left_pad=False)
    segment_ids = merge('segment_ids', left_pad=False)
    lm_label_ids = merge('lm_label_ids', left_pad=False)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'masked_lm_labels': lm_label_ids,
            'output_mask': lm_label_ids != pad_idx,
        },
        'target': lm_label_ids,
    }

    return batch


class ConditionalMLMDataset(FairseqDataset):

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        cmlm_alpha=0.15, conditional=True, source_da=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.cmlm_alpha = cmlm_alpha
        self.conditional = conditional
        self.pad_idx = tgt_dict.pad()
        self.source_da = source_da

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        raw_src = self.src_dict.string(src_item)
        raw_tgt = self.tgt_dict.string(tgt_item)

        # tokens_b + tokens_a
        if self.source_da:
            tokens_a = raw_tgt.split()
            tokens_b = raw_src.split()
        else:
            tokens_a = raw_src.split()
            tokens_b = raw_tgt.split()


        #while True and self.conditional:
        #    tokens_a, t1_label = random_word(tokens_a, self.tgt_dict)
        #    if any(label != -1 for label in t1_label):
        #        break
        tokens_a = tokens_a + [SEP]
        t1_label = [-1] * len(tokens_a)

        while True:
            tokens_b, t2_label = random_word(tokens_b, self.tgt_dict)
            if any(label != -1 for label in t2_label):
                break

        tokens = []
        segment_ids = []
        tokens.append(CLS)
        segment_ids.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(0)

        if not self.conditional:
            lm_label_ids = [-1] + t2_label
        else:
            lm_label_ids = ([-1] + t2_label + t1_label)

        if self.conditional:
            assert len(tokens_a) > 0
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(1)
        # tokens.append(SEP)
        # segment_ids.append(1)
        # NOTE the last SEP is handled differently from original BERT

        input_ids = [self.src_dict.index(tok) for tok in tokens]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to multiples of 8 (for tensor cores)
        while len(input_ids) % 8 != 0:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) % 8 == 0
        assert (len(input_ids) == len(input_mask)
                == len(segment_ids) == len(lm_label_ids))

        lm_label_ids = [label_id if label_id != -1 else self.pad_idx for label_id in lm_label_ids]

        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.LongTensor(input_mask)
        segment_ids = torch.LongTensor(segment_ids)
        lm_label_ids = torch.LongTensor(lm_label_ids)
        return {
            'id': index,
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'lm_label_ids': lm_label_ids,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        return collate(samples, self.src_dict.pad(), self.src_dict.eos())

    def size(self, index):
        return (self.src_sizes[index], self.tgt_sizes[index])

    def num_tokens(self, index):
        return max(self.src_sizes[index], self.tgt_sizes[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
