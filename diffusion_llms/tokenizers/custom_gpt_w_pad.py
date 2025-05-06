import tiktoken

class CustomGPT2TokenizerWithPad:
    def __init__(self, pretrained_model_name='gpt2'):
        # Load GPT-2 encoder
        self.tokenizer = tiktoken.get_encoding(pretrained_model_name)
        # Add <pad> token if not present
        if '<pad>' not in self.tokenizer.special_tokens_set:
            # Assign a new token ID for <pad>
            self.pad_token = '<pad>'
            self.pad_token_id = max(self.tokenizer._special_tokens.values(), default=50256) + 1
            self.tokenizer._special_tokens[self.pad_token] = self.pad_token_id
            self.tokenizer.special_tokens_set.add(self.pad_token)
            self.tokenizer._special_tokens[self.pad_token] = self.pad_token_id
            #self.tokenizer._id_to_special_token[self.pad_token_id] = self.pad_token
        else:
            self.pad_token = '<pad>'
            self.pad_token_id = self.tokenizer._special_tokens['<pad>']

    def encode(self, text, max_length=None, padding=False, return_tensors=None):
        # Encode text and pad if required
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if padding and max_length is not None:
            pad_len = max_length - len(encoded)
            if pad_len > 0:
                encoded = encoded + [self.pad_token_id] * pad_len
            else:
                encoded = encoded[:max_length]
        if return_tensors == 'pt':
            import torch
            return torch.tensor([encoded])
        return encoded

    def decode(self, token_ids, skip_special_tokens=True):
        # Remove pad tokens before decoding
        if isinstance(token_ids, list):
            token_ids = [tid for tid in token_ids if tid != self.pad_token_id]
        elif hasattr(token_ids, 'tolist'):
            token_ids = [tid for tid in token_ids.tolist() if tid != self.pad_token_id]
        return self.tokenizer.decode(token_ids)

    def apply_template(self, sentence, template_type=1, max_length=None, padding=False):
        """
        Apply a template to the sentence and return the tokenized sequence.

        Automatically adds padding tokens to the end of the sequence if max_length is specified.

        template_type:
            1: eos + sentence + eos
            2: sentence + eos
        """
        eos_token = "<|endoftext|>"
        eos = eos_token if eos_token else ''
        # Build template string
        if template_type == 1: # eos + sentence + eos + --- + eos
            template = f"{eos} {sentence} {eos}"
        elif template_type == 2: # sentence + eos + --- + eos
            template = f"{sentence} {eos}"
        else:
            raise ValueError("Invalid template_type. Must be 1 or 2")
        # Tokenize
        tokens = self.encode(template, max_length=max_length, padding=padding)
        return tokens