import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from models.decoder import TransformerDecoder


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class BertSummarizer(nn.Module):
    def __init__(self, checkpoint, device, temp_dir='/temp'):
        super(BertSummarizer, self).__init__()
        self.device = device
        self.bert = Bert(False, temp_dir, True)

        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)

        self.decoder = TransformerDecoder(6, 768, heads=8, d_ff=2048, dropout=0.2, embeddings=tgt_embeddings)
        self.generator = get_generator(self.vocab_size, 768, self.device)
        self.generator[0].weight = self.decoder.embeddings.weight

        self.load_state_dict(checkpoint['model'], strict=True)
        self.to(self.device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
