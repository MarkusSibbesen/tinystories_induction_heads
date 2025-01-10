from nltk.tag import pos_tag
from nltk.tokenize import TreebankWordTokenizer

class PosTagger:

    def __init__(self, tokenizer):
        self.treebank_tokenizer = TreebankWordTokenizer()
        self.tokenizer = tokenizer

    def make_charlvl_tags(self, text):
        tags_li = [''] * len(text)

        treebank_tokenizer = TreebankWordTokenizer()
        tokens = treebank_tokenizer.tokenize(text)
        pos_tags = pos_tag(tokens)
        spans = list(treebank_tokenizer.span_tokenize(text))

        for span, (token, tag) in zip(spans, pos_tags):
            for i in range(*span):
                tags_li[i] = tag

        return tags_li
    

    def tag_input(self, text, return_words=False):
        '''
        return tokens, tags
        '''
        char_lvl_tags = self.make_charlvl_tags(text)
        tokens = self.tokenizer(text, return_offsets_mapping=True)

        tags = []
        for start, stop in tokens['offset_mapping']:
            if (stop - start) % 2 == 0:
                avg_idx = (start + stop) // 2
                tags.append(char_lvl_tags[avg_idx])
            else:
                avg_idx = ((start + stop) // 2) + 1 # round up
                avg_idx = min(avg_idx, len(text) - 1) # handle idx out of bound wh rounding up
                tags.append(char_lvl_tags[avg_idx])

        if return_words:
            return tokens, tags, [self.tokenizer.decode(token) for token in tokens['input_ids']]
        
        return tokens, tags