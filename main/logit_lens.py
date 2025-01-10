import torch
from attention_extraction import get_head_outparams



def logit_lens(model, tokenizer, vector, layer, head, k):

    unembedding_weight = model.get_submodule('lm_head').weight.detach().numpy()

    weight, bias = get_head_outparams(model, layer, head)

    token_logits = unembedding_weight @ (weight @ vector)

    indices = torch.topk(torch.Tensor(token_logits), k).indices
    decoded = tokenizer.decode(indices)

    return decoded






if __name__ == '__main__':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from pos_tagger import PosTagger
    from data_handling import load_tinystories_data
    from matplotlib import pyplot as plt

    from attention_extraction import get_head_out, get_head_outparams, extract_all_attention, get_causal_selfattention_pattern, plot_idx_of_highest_output
    from plotting import plot_selfattention_from_idx

    data = load_tinystories_data('data/tinystories_val.txt')

    model_url = 'roneneldan/TinyStories-1M'

    model = AutoModelForCausalLM.from_pretrained(model_url, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    keys, queries, values = extract_all_attention(model, tokenizer, data[0])
    attention = get_causal_selfattention_pattern(keys[0][0], queries[0][0])

    pos_tagger = PosTagger(tokenizer)
    tokens, tags, words = pos_tagger.tag_input(data[0], return_words=True)




