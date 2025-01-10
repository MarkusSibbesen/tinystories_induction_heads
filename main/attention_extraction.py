import torch
from torch import Tensor
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt


class HookManager():
    def __init__(self, model):
        self.model = model
        self.num_heads = self.model.config.num_heads
        self.hooks = []

    
    def attach_attention_hook(self, layer, vector):
        match vector:
            case 'value': hookpoint = f'transformer.h.{layer}.attn.attention.v_proj'
            case 'key': hookpoint = f'transformer.h.{layer}.attn.attention.k_proj'
            case 'query': hookpoint = f'transformer.h.{layer}.attn.attention.q_proj'
            case _: raise ValueError("vector has to be from {'value', 'key', query}")

        extracted_attention = []
        def attention_hook(module, input, output):
            extracted_attention.append(
                torch.stack(torch.chunk(output.squeeze(0).detach(), self.num_heads, dim=-1))
            )

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(attention_hook)
        )

        return extracted_attention


    def attach_outproj_input_hook(self, layer):

        hookpoint = f'transformer.h.{layer}.attn.attention.out_proj'

        extracted_input = []
        def attention_hook(module, input, output):
            extracted_input.append(input[0].squeeze(0))

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(attention_hook)
        )

        return extracted_input
    

    def attach_outproj_output_hook(self, layer):

        hookpoint = f'transformer.h.{layer}.attn.attention.out_proj'

        extracted_output = []
        def attention_hook(module, input, output):
            extracted_output.append(output[0].squeeze(0))

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(attention_hook)
        )

        return extracted_output


    
    def __enter__(self):
        return self
    
    def __exit__(self,exc_type, exc_value, traceback):

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()




def extract_all_attention(model, tokenizer, input: list[str] | str):
    '''
    returns keys, queries, values
    '''

    if type(input) == str:
        input = [input]

    num_layers = model.config.num_layers
    keys = []
    queries = []
    values = []

    with HookManager(model) as hook_manager:
        for layer in range(num_layers):
            keys.append(hook_manager.attach_attention_hook(layer, 'key'))
            queries.append(hook_manager.attach_attention_hook(layer, 'query'))
            values.append(hook_manager.attach_attention_hook(layer, 'value'))

        for input_ in input:
            tokenized = tokenizer.encode(input_, return_tensors='pt')
            out = model.forward(tokenized)

    keys = torch.stack(
        [torch.concat(layer, dim=1) for layer in keys]
    ).numpy()
    queries = torch.stack(
        [torch.concat(layer, dim=1) for layer in queries]
    ).numpy()
    values = torch.stack(
        [torch.concat(layer, dim=1) for layer in values]
    ).numpy()

    return keys, queries, values



def get_causal_selfattention_pattern(keys, queries):
    '''
    takes Tensors (or arrays) for one head:
        keys of shape [num_tokens, head_dim]
        queries of shape [num_tokens, head_dim]
    outputs attention pattern for that head

    *this can also be sone easier by calling outputs.attentions*
    '''

    att_pattern = torch.Tensor(queries @ keys.T)
    mask = torch.triu(torch.ones_like(att_pattern), diagonal=1).bool()
    att_pattern[mask] = float('-inf')

    att_pattern = torch.softmax(att_pattern, dim=1)

    return att_pattern.numpy()



def get_head_outparams(model, layer, head):

    head_dim = model.config.hidden_size // model.config.num_heads
    out_proj = model.get_submodule(f'transformer.h.{layer}.attn.attention.out_proj')

    weight = out_proj.weight[:, head * head_dim: head * head_dim + head_dim].detach().numpy()
    bias = out_proj.bias.detach().numpy()

    return weight, bias



def get_head_out(model, tokenizer, layer, head, input):

    head_dim = model.config.hidden_size // model.config.num_heads

    if type(input) == str:
        input = [input]

    with HookManager(model) as hook_manager:

        outs = hook_manager.attach_outproj_input_hook(layer)

        for input_ in input:
            tokenized = tokenizer.encode(input_, return_tensors='pt')
            model.forward(tokenized)

    outs = torch.concat(outs, dim=0).detach().numpy()

    return outs[:, head * head_dim : head * head_dim + head_dim]



def get_head_norms_out(model, tokenizer, layer, head, input):

    weight, bias = get_head_outparams(model, layer, head)
    outs = get_head_out(model, tokenizer, layer, head, input)

    norms = [
        np.linalg.norm(weight @ out).item()
        for out in outs
    ]

    return norms




if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from pos_tagger import PosTagger
    from data_handling import load_tinystories_data
    from matplotlib import pyplot as plt

    data = load_tinystories_data('data/tinystories_val.txt')

    model_url = 'roneneldan/TinyStories-1M'

    model = AutoModelForCausalLM.from_pretrained(model_url, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    pos_tagger = PosTagger(tokenizer)
    inputs = ['this is a sentence, too', 'this too']

    keys, queries, values = extract_all_attention(model, tokenizer, inputs[0])

    attention = get_causal_selfattention_pattern(keys[0][0], queries[0][0])

    tokens, tags = pos_tagger.tag_input(inputs[0])

    

