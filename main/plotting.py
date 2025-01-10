from matplotlib import pyplot as plt
import numpy as np
from attention_extraction import get_head_norms_out
from collections import Counter

def plot_selfattention(attention, words, tags):
       
    fig, axs = plt.subplots(len(words), figsize=(5, len(words)*2))
    axs = axs.flatten()


    for i in range(len(words)):
        axs[i].set_title(words[i])

        barplot = [attention[i][j].item() for j in range(len(words))]

        axs[i].bar(range(len(words)), barplot)
        axs[i].set_xticks(range(len(words)))
        axs[i].set_xticklabels([f'{word} ({tag})' for word, tag in zip(words, tags)], rotation=45, ha='right')


    fig.tight_layout()
    plt.show()


def plot_selfattention_from_idx(attention, words, tags, idx, context_size, color='blue', outfile=None):

    context_size = min(idx + 1, context_size)

    attn_context = attention[idx, max(0, idx + 1 - context_size) : idx + 1]
    words_context = words[max(0, idx + 1 - context_size) : idx + 1]
    tags_context = tags[max(0, idx + 1 - context_size) : idx + 1]


    fig, axs = plt.subplots(1, figsize=(12, 4))

    axs.set_title(f'Attention from{words[idx]} ({tags[idx]})')

    barplot = [attn_context[j].item() for j in range(context_size)]

    axs.bar(range(context_size), barplot,color='darkblue')
    axs.set_xticks(range(context_size))
    axs.set_xticklabels([f'{word} ({tag})' for word, tag in zip(words_context, tags_context)], rotation=90, ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylabel("Attention")



    fig.tight_layout()
    plt.show()

    if outfile != None:
        fig.savefig(outfile, bbox_inches='tight')

    return fig, axs



def plot_probe_results_from_tag(results, tag, outfile=None, cmap='viridis'):
    fig, axs = plt.subplots(1, figsize=(7, 3))

    num_layers = len(results)
    num_heads = len(results[0])

    field = [
        [
            results[layer][head][tag]['f1-score'] if tag in results[layer][head] else 0.0
            for head in range(num_heads)
        ]
        for layer in range(num_layers)
    ]


    field = np.array(field)

    x = np.arange(field.shape[1] + 1)
    y = np.arange(field.shape[0] + 1)

    c = axs.pcolormesh(x, y, field, shading='flat', vmin=0, vmax=1, cmap=cmap)


    # Set ticks at the center of squares
    axs.set_xticks(np.arange(field.shape[1]) + 0.5)
    axs.set_yticks(np.arange(field.shape[0]) + 0.5)
    
    # Set tick labels
    axs.set_xticklabels(range(field.shape[1]))
    axs.set_yticklabels(range(field.shape[0]))

    axs.set_ylabel('Layer')
    axs.set_xlabel('Head')
    axs.set_title(tag)
    
    fig.colorbar(c, label="F1-score", ax=axs)

    plt.show()

    if outfile != None:
        fig.savefig(outfile, bbox_inches='tight')



def plot_idx_of_highest_output(model, tokenizer, input, layer, head, pos_tagger, outfile=None, color='blue'):
    fig, ax = plt.subplots(1, figsize=(8,2.5))

    tokens, tags, words = pos_tagger.tag_input(input, return_words='True')
    head_norms = get_head_norms_out(model, tokenizer, layer, head, input)

    top = np.argmax(head_norms)
    low = max(0, top-15)
    high = min(len(words), top+15)
    len_graph = high - low

    ax.bar(range(len_graph), head_norms[low:high], color=color)
    ax.set_xticks(range(len_graph))
    ax.set_xticklabels([f'{word} ({tag})' for word, tag in zip(words[low:high], tags[low:high])], rotation=60, ha='right', rotation_mode='anchor')

    ax.set_ylabel('Norm of vector')
    plt.show()

    if outfile != None:
        fig.savefig(outfile, bbox_inches='tight')

    return top, fig, ax


def plot_activity(model, tokenizer, input, layer, head, idx, context, pos_tagger, outfile=None, color='blue'):
    fig, ax = plt.subplots(1, figsize=(8,3))

    top = idx

    tokens, tags, words = pos_tagger.tag_input(input, return_words='True')
    head_norms = get_head_norms_out(model, tokenizer, layer, head, input)
    low = max(0, top-context)
    high = min(len(words), top+context)
    len_graph = high - low

    ax.bar(range(len_graph), head_norms[low:high], color=color)
    ax.set_xticks(range(len_graph))
    ax.set_xticklabels(words[low:high], rotation=60, ha='right', rotation_mode='anchor')

    ax.set_ylabel('Norm of vector')
    plt.show()

    if outfile != None:
        fig.savefig(outfile)

def plot_f1_matrix(results:list,filename:str,color:str):
    tags = list(results[0][0].keys())
    labels = tags[:tags.index('accuracy')]
    num_labels = len(labels)

    num_layers = len(results)
    num_heads = len(results[0])

    fig, axs = plt.subplots(8, 5, figsize=(15, 15))
    axs = axs.flatten()

    for idx, label in enumerate(labels):

        label = str(label)

        field = [
            [
                results[layer][head][label]['f1-score'] if label in results[layer][head] else 0.0
                for head in range(num_heads)
            ]
            for layer in range(num_layers)
        ]


        field = np.array(field)

        x = np.arange(field.shape[1] + 1)
        y = np.arange(field.shape[0] + 1)

        c = axs[idx].pcolormesh(x, y, field, cmap=color, shading='flat', vmin=0, vmax=1)
        axs[idx].set_title(f'POS-Tag: {label}') #counts={counts[label] if label in counts else 0}')
        
        fig.colorbar(c, label="Intensity", ax=axs[idx])

    for extra_ax in axs[num_labels:]:
        extra_ax.axis('off')  
    
    fig.tight_layout()

    fig.savefig(f'probes_{filename}.pdf')
    plt.show()


if __name__ == '__main__':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from pos_tagger import PosTagger
    from attention_extraction import extract_all_attention, get_causal_selfattention_pattern

    from data_handling import load_tinystories_data

    data = load_tinystories_data('data/tinystories_val.txt')


    model_url = 'roneneldan/TinyStories-1M'

    model = AutoModelForCausalLM.from_pretrained(model_url, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_url)

    pos_tagger = PosTagger(tokenizer)
    inputs = data

    keys, queries, values = extract_all_attention(model, tokenizer, inputs[0])

    tokens, tags, words= pos_tagger.tag_input(inputs[0], return_words=True) 

    attention = get_causal_selfattention_pattern(keys[0][0], queries[0][0])