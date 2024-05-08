'''
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import numpy as np
import torch
from functools import lru_cache
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import re

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_wikitext2(n_samples, seed, seqlen, model):
    print("get_wikitext2", flush=True)
    from datasets import load_dataset
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    print("get_wikitext2 testenc", flush=True)
    test_enc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    # test_enc = tokenizer("\n\n".join(testdata['text'][:n_samples]), return_tensors='pt')
    print("get_wikitext2 test_enc", test_enc, flush=True)

    return test_enc

def get_wikitext_de(n_samples, seed, seqlen, model):
    print("Fetching WikiText-DE dataset from Hugging Face Hub", flush=True)

    # Load the dataset
    dataset_path = "LeoLM/wikitext-en-de"
    dataset_name = "exzellent_de_small"
    dataset = load_dataset(dataset_path, dataset_name, split='train')

    # Seed numpy's random generator for reproducibility in sampling
    np.random.seed(seed)

    # If n_samples is more than the dataset size, adjust it to the dataset size
    n_samples = min(n_samples, len(dataset))

    # Randomly sample n_samples indices from the dataset
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    sampled_dataset = dataset.select(indices)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use the eos token if no pad token is available

    # Process texts to fit within seqlen
    tokenized_texts = []
    for entry in sampled_dataset:
        text = entry['text']
        # Split text into segments that are likely to be under the seqlen limit
        segments = []
        current_segment = ""
        for sentence in re.split(r'(?<=[.!?])\s+', text):  # Split text into sentences
            if len(tokenizer.encode(current_segment + sentence, add_special_tokens=False)) > seqlen:
                if current_segment:  # Store the current segment if it's not empty
                    segments.append(current_segment)
                current_segment = sentence
            else:
                current_segment += (" " + sentence if current_segment else sentence)
        if current_segment:  # Don't forget to add the last segment
            segments.append(current_segment)

        # Tokenize each segment separately
        for segment in segments:
            encoded_text = tokenizer(segment, max_length=seqlen, truncation=True, padding="max_length", return_tensors='pt')
            tokenized_texts.append(encoded_text)

    return tokenized_texts

@lru_cache
def get_ptb(n_samples, seed, seqlen, model):
    print("get_ptb", flush=True)
    from datasets import load_dataset
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    test_enc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    # test_enc = tokenizer(" ".join(testdata['sentence'][:n_samples]), return_tensors='pt')

    print("get_ptb testenc", flush=True)
    
    return test_enc

@lru_cache
def get_c4(n_samples, seed, seqlen, model):
    print("get_c4", flush=True)
    from datasets import load_dataset
    data_files = {"validation": "en/c4-validation.00000-of-00008.json.gz"}
    val_data = load_dataset("allenai/c4", data_files=data_files, split="validation")
    
    print("get_c4 testenc downloaded",val_data, flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    import random
    random.seed(seed)

    val_enc = tokenizer(' '.join(val_data[:8192]['text']), return_tensors='pt')
    # val_enc = tokenizer(' '.join(val_data[:n_samples]['text']), return_tensors='pt')
    val_enc = val_enc.input_ids[:, :(256 * seqlen)]
    print("get_c4 testenc wrapped", flush=True)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    val_enc = TokenizerWrapper(val_enc)
    print("get_c4 testenc", flush=True)

    return val_enc


@lru_cache
def get_gsm8k(n_samples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random

    # load gsm8k dataset
    testdata = load_dataset('gsm8k', 'main', split='test')

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # set seed
    random.seed(seed)

    # prepare dataset for test
    test_tokens = []
    for i in range(n_samples):
        encoded = tokenizer(testdata[i]['question'], return_tensors='pt', padding="max_length", truncation=True, max_length=seqlen)
        test_tokens.append(encoded.input_ids)

    # converting the list of tensors into a single tensor and wrapping it in a dictionary
    test_tokens_tensor = torch.cat(test_tokens, dim=0)
    test_data_dict = {'input_ids': test_tokens_tensor}

    return test_data_dict

@lru_cache
def get_test_tokens(
    name, n_samples = 256, seed=0, seqlen=2048, model=''
):
    if name == 'wikitext2':
        return get_wikitext2(n_samples, seed, seqlen, model)['input_ids']
    elif name == 'wikitext_de':
        return get_wikitext_de(n_samples, seed, seqlen, model)["input_ids"]
    elif name == 'ptb':
        return get_ptb(n_samples, seed, seqlen, model)["input_ids"]
    elif name == 'c4':
        return get_c4(n_samples, seed, seqlen, model).input_ids
    elif name == 'gsm8k':
        return get_gsm8k(n_samples, seed, seqlen, model)['input_ids']
    else:
        raise Exception
