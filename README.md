# BERT Summarizer

Text Summarization with Pretrained Encoders

This repository is built from the [PreSumm](https://github.com/nlpyang/PreSumm) repository by [nlpyang](https://github.com/nlpyang). I created this repo for people who just need a plug-and-play implementation of the summarization model that is ready to be integrated into any ml pipeline.

The pretrained BertSumExtAbs model (1.98GB) will download automatically from gdrive. Import the `AbstractSummarizer` class or use the Flask app.

## Sample Usage

```python
summarizer = AbstractSummarizer()
print(summarizer.get_summary([
    r"""Shares of Starbucks Corp. SBUX sank 2.1% in premarket trading Friday, putting them on track to open at an 11-month low, amid growing concerns over the impact of the coronavirus outbreak. Analyst Nick Setyan at Wedbush cut his price target by 12% to $84 from $95, while maintaining the neutral rating he's had on the coffee giant for the past two years. Setyan said that while the coronavirus impact in China appears to be "bracketed," the rest of the world, including the U.S., is at risk. "While management noted U.S. same-store sales growth momentum continues with no evidence of COVID-19 impact to-date, we believe forward expectations are somewhat at risk," Setyan wrote in a note to clients. "Should a disruption ensue in the near term, we estimate that a 10% sales decline per week would impact EPS by [1 cent-2 cents] per week." The stock has dropped 11.7% over the past three months through Thursday, while the S&P 500 SPX, -2.22% has lost 3.9%.""",
    r"""A banana is an edible fruit – botanically a berry[1][2] – produced by several kinds of large herbaceous flowering plants in the genus Musa.[3] In some countries, bananas used for cooking may be called "plantains", distinguishing them from dessert bananas. The fruit is variable in size, color, and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a rind, which may be green, yellow, red, purple, or brown when ripe. The fruits grow in clusters hanging from the top of the plant. Almost all modern edible seedless (parthenocarp) bananas come from two wild species – Musa acuminata and Musa balbisiana. The scientific names of most cultivated bananas are Musa acuminata, Musa balbisiana, and Musa × paradisiaca for the hybrid Musa acuminata × M. balbisiana, depending on their genomic constitution. The old scientific name for this hybrid, Musa sapientum, is no longer used. Musa species are native to tropical Indomalaya and Australia, and are likely to have been first domesticated in Papua New Guinea.[4][5] They are grown in 135 countries,[6] primarily for their fruit, and to a lesser extent to make fiber, banana wine, and banana beer and as ornamental plants. The world's largest producers of bananas in 2017 were India and China, which together accounted for approximately 38% of total production"""
]))

>>> ['shares of starbucks corp. sbux sank 2.1 in premarket trading friday<q>putting them on track to open at an 11-month low<q>amid growing concerns over the impact of the coronavirus outbreak', 'a banana is an edible fruit<q>in some countries, bananas used for cooking may be called "plantains"']
```
