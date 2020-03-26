import torch
from pytorch_transformers import BertTokenizer
from utils.utils import load_text
from models.model_builder import BertSummarizer
from models.predictor import build_predictor
import os
import glob
import gdown

# must be gdrive sharable link
MODEL_URL = 'https://drive.google.com/a/boastcapital.com/uc?id=1-IKVCtc4Q-BdZpjXc4s70_fRsWnjtYLr'

class AbstractSummarizer(object):
    def __init__(self, model_path='cache/abs_bert_model.pt'):
        if not os.path.exists('cache'):
            os.mkdir('cache')

        # check if model is downloaded
        if not os.path.exists(model_path):
            print('Model not found in cache')
            self.download_model(model_path)

        # setup cache for bert model and tokenizer
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertSummarizer(checkpoint, device, cache_dir)
        self.model.eval()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=cache_dir)
        self.predictor = build_predictor(tokenizer, self.model)

    def get_summary(self, texts):
        text_iter = load_text(texts)
        return self.predictor.translate(text_iter)

    @staticmethod
    def download_model(model_path):
        # download
        temp = 'cache/temp.zip'
        gdown.download(MODEL_URL, temp, quiet=False)

        # unzip
        print('unzipping file')
        gdown.extractall(temp)
        os.remove(temp)

        # rename
        os.rename(glob.glob('cache/*.pt')[0], model_path)


if __name__ == '__main__':
    summarizer = AbstractSummarizer()
    print(summarizer.get_summary([
        r"""Shares of Starbucks Corp. SBUX sank 2.1% in premarket trading Friday, putting them on track to open at an 11-month low, amid growing concerns over the impact of the coronavirus outbreak. Analyst Nick Setyan at Wedbush cut his price target by 12% to $84 from $95, while maintaining the neutral rating he's had on the coffee giant for the past two years. Setyan said that while the coronavirus impact in China appears to be "bracketed," the rest of the world, including the U.S., is at risk. "While management noted U.S. same-store sales growth momentum continues with no evidence of COVID-19 impact to-date, we believe forward expectations are somewhat at risk," Setyan wrote in a note to clients. "Should a disruption ensue in the near term, we estimate that a 10% sales decline per week would impact EPS by [1 cent-2 cents] per week." The stock has dropped 11.7% over the past three months through Thursday, while the S&P 500 SPX, -2.22% has lost 3.9%.""",
        r"""A banana is an edible fruit – botanically a berry[1][2] – produced by several kinds of large herbaceous flowering plants in the genus Musa.[3] In some countries, bananas used for cooking may be called "plantains", distinguishing them from dessert bananas. The fruit is variable in size, color, and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a rind, which may be green, yellow, red, purple, or brown when ripe. The fruits grow in clusters hanging from the top of the plant. Almost all modern edible seedless (parthenocarp) bananas come from two wild species – Musa acuminata and Musa balbisiana. The scientific names of most cultivated bananas are Musa acuminata, Musa balbisiana, and Musa × paradisiaca for the hybrid Musa acuminata × M. balbisiana, depending on their genomic constitution. The old scientific name for this hybrid, Musa sapientum, is no longer used. Musa species are native to tropical Indomalaya and Australia, and are likely to have been first domesticated in Papua New Guinea.[4][5] They are grown in 135 countries,[6] primarily for their fruit, and to a lesser extent to make fiber, banana wine, and banana beer and as ornamental plants. The world's largest producers of bananas in 2017 were India and China, which together accounted for approximately 38% of total production"""
    ]))
