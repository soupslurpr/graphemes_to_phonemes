# usage:
# python thisscript.py train
# python thisscript.py run Example

import os
import json
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)
import numpy as np
import sys
import torch
import re

british = True
if british:
    gold_dataset_path = "./gb_gold.json"
    silver_dataset_path = "./gb_silver.json"
    model_dir = "./en_gb_model"
else:
    gold_dataset_path = "./us_gold.json" 
    silver_dataset_path = "./us_silver.json" 
    model_dir = "./en_us_model"

# This is designed to load the datasets from `misaki`.
def load_dataset(path):
    with open(path) as f: 
        j = json.loads(f.read())
    dataset = []
    for english, phonemes in j.items():
        if type(phonemes) == str:
            dataset.append((english, phonemes))
    return dataset

# Convert a grapheme `word`, if it is plural, to its phonemes.
# The requires that its singular pronounciation be in phoneme_lookup and that it follows "regular" pluralization rules.
# (Irregular plurals are presumed to already be in the dataset.)
def get_plural_phonemes(word, phoneme_lookup):
    if len(word) < 3 or not word.endswith('s'):
        return None
    if not word.endswith('ss'):
        stem = word[:-1]
    elif (word.endswith("'s") or (len(word) > 4 and word.endswith('es') and not word.endswith('ies'))):
        stem = word[:-2]
    elif len(word) > 4 and word.endswith('ies'):
        stem = word[:-3] + 'y'
    else:
        return None
    stem_phonemes = phoneme_lookup.get(stem, None)
    if not stem_phonemes: return None

    if stem[-1] in 'ptkfθ':
        return stem_phonemes + 's'
    elif stem[-1] in 'szʃʒʧʤ':
        return stem_phonemes + ('ɪ' if british else 'ᵻ') + 'z'
    return stem_phonemes + 'z'

# The dataset does not have regular plural versions of words, but the model does need
# to be able to handle that. To get examples of plural words that are actually used,
# we ingest a large text file and check if each word is a plural of a known-good word.
# If so, we can derive is pronounciation by following a simple rule.
# (There are some plural versions in the dataset, but they are mostly the unusual ones.)
def augment_dataset(dataset, text_file):
    phoneme_lookup = {graphemes: phonemes for (graphemes, phonemes) in dataset}
    added = set()
    extra_dataset = []
    with open(text_file) as f:
        for line in f.readlines():
            words = re.findall(r'\b\w+\b', line)
            for word in words:
                word = word.lower()
                if word in phoneme_lookup or word in added: continue

                plural_phonemes = get_plural_phonemes(word, phoneme_lookup)
                if plural_phonemes is not None:
                    extra_dataset.append((word, plural_phonemes))
                    added.add(word)
    return extra_dataset

def vocab_for_data(dataset):
    grapheme_chars = set()
    phoneme_chars = set()
    for graphemes, phonemes in dataset:
        grapheme_chars.update(list(graphemes))
        phoneme_chars.update(list(phonemes))
    print(grapheme_chars)
    print(phoneme_chars)
    # To have an clear char -> token id mapping, the characters that occur
    # as both graphemes and phonemes will be identically ordered at the
    # start for both.
    chars_in_common = set(grapheme_chars).intersection(set(phoneme_chars))
    only_grapheme = set(grapheme_chars).difference(chars_in_common)
    only_phoneme = set(phoneme_chars).difference(chars_in_common)

    grapheme_chars = sorted(list(chars_in_common)) + sorted(list(only_grapheme))
    phoneme_chars = sorted(list(chars_in_common)) + sorted(list(only_phoneme))
    return grapheme_chars, phoneme_chars

# This model uses character-level tokenization. The only oddity is that, since the
# input and output characters come from different sets, we re-use token indices between
# the two.
class CharTokenizer(PreTrainedTokenizer):
    """
    A very simple character-level tokenizer.
    """
    def __init__(self, grapheme_chars, phoneme_chars):
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        # Reserve first few indices for special tokens.
        # We will use:
        #   <pad>: 0, <s>: 1, </s>: 2, <unk>: 3.
        special_vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        grapheme_vocab_list = special_vocab + grapheme_chars
        phoneme_vocab_list = special_vocab + phoneme_chars
        print(grapheme_vocab_list)
        print(phoneme_vocab_list)
        
        vocab = {token: idx for idx, token in enumerate(grapheme_vocab_list)}
        vocab.update({token: idx for idx, token in enumerate(phoneme_vocab_list)})
        self.vocab = vocab
        # Build an inverse mapping from id to token.
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        self.ids_to_phonemes = {i: t for i, t in enumerate(phoneme_vocab_list)}
        self.ids_to_graphemes = {i: t for i, t in enumerate(grapheme_vocab_list)}
        super().__init__()
    
    def _tokenize(self, text):
        # Split text into characters.
        return list(text)
    
    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get("<unk>"))
    
    def convert_tokens_to_string(self, tokens):
        # Rejoin tokens as a string.
        return "".join(tokens)
   
    def decode_phonemes(self, tokens):
        return "".join([self.ids_to_phonemes.get(token, "<unk>") for token in tokens.tolist() if token > 3])

    def encode(self, text):
        return [1] + [self.vocab.get(c, 3) for c in text] + [2]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # For a single sentence, add bos at beginning and eos at end.
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # For sequence pair, you can define your own method.
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    @property
    def vocab_size(self):
        return 4 + max(len(self.ids_to_phonemes), len(self.ids_to_graphemes))
    def get_vocab(self):
        return self.vocab
    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def encode_example(example):
    # Each example is a tuple: (input_text, target_text)
    source, target = example
    # Encode input and target with special tokens
    source_ids = tokenizer.encode(source)
    target_ids = tokenizer.encode(target)
    return {"input_ids": source_ids, "labels": target_ids}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = sys.argv[1]
if mode == 'train':
    gold_data = load_dataset(gold_dataset_path)
    silver_data = load_dataset(silver_dataset_path)
    grapheme_chars, phoneme_chars = vocab_for_data(gold_data + silver_data)
    training_data = gold_data + [d for i, d in enumerate(silver_data) if i%10 != 0]
    eval_data = [d for i, d in enumerate(silver_data) if i%10 == 0]
    training_data += augment_dataset(training_data, '../wiki.train.raw')

    print(grapheme_chars, phoneme_chars)
    print(training_data[-20:])
    tokenizer = CharTokenizer(grapheme_chars, phoneme_chars)
    
    # Tokenize all samples
    dataset = Dataset.from_list([encode_example(pair) for pair in training_data])
    eval_dataset = Dataset.from_list([encode_example(pair) for pair in eval_data])

    # We train a BartForConditionalGeneration from scratch.
    # These parameters have been pretty vigorously minimized to produce a small model
    # that still works properly.
    vocab_size = tokenizer.vocab_size
    config = BartConfig(
        vocab_size=vocab_size,
        d_model=128,                 # hidden size
        encoder_ffn_dim = 1024,
        decoder_ffn_dim = 1024,
        encoder_layers=1,           # number of encoder layers
        decoder_layers=1,           # number of decoder layers
        encoder_attention_heads=1,  # attention heads in encoder
        decoder_attention_heads=1,  # attention heads in decoder
        decoder_start_token_id=tokenizer.bos_token_id,
        max_position_embeddings = 64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.grapheme_chars = "____" + "".join(grapheme_chars)
    config.phoneme_chars = "____" + "".join(phoneme_chars)
    model = BartForConditionalGeneration(config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        learning_rate=5e-4,
        logging_steps=100,
        save_strategy = "no", # No need for checkpoint saving, since this is so small
        predict_with_generate=True,
        eval_strategy = "epoch",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset = eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()

    model.save_pretrained(model_dir)
    
elif mode == 'run':
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    # grapheme_chars and phoneme_chars start with "____" for the convience of the runtime version of this.
    tokenizer = CharTokenizer(list(model.config.grapheme_chars.lstrip('_')), list(model.config.phoneme_chars.lstrip('_')))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    model.to(device)
    model.eval()

    input_text = sys.argv[2]
    print("Running on", input_text)
    input_ids = torch.tensor([tokenizer.encode(input_text)], device = device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids = input_ids)

    output_text = tokenizer.decode_phonemes(generated_ids[0])
    print(f"Phonemes for '{input_text}':", output_text)
