# Zidong
About 5900
1 Loading EmoBank
EmoBank comes with annotations according to two perspectives (reader and writer). However, for most use cases, this distinction may not be relevant. In these cases, I would advise to use the combination of both annotions perspectives to increase reliability. These combined ratings are stored for convenience in emobank.csv and can be loaded like this:

import pandas as pd
eb = pd.read_csv("emobank.csv", index_col=0)

1.1 Data Format
The columns V, A and D represent Valence (negative vs. positive), Arousal (calm vs. excited), and Dominance (being controlled vs. being in control). Each of those take numeric values from [1, 5]. Please refer to the paper for further details.



print(eb.shape)
eb.head()

# Overview
This repository contains EmoBank, a large-scale text corpus manually annotated with emotion according to the psychological Valence-Arousal-Dominance scheme. It was build at JULIE Lab, Jena University and is described in detail in our papers from EACL 2017 and LAW 2017 (see Citation). The repository contains two folders: "corpus" which contains the actual Emobank data (described in the EACL paper) and "pilot" which contains the data from our pilot study (described in the LAW paper). See the readme files in the respective folders for more detailed information regarding the data format.

# References
Nancy C. Ide, Collin F. Baker, Christiane Fellbaum, and Rebecca J. Passonneau. 2010. The Manually Annotated Sub-Corpus: A community resource for and by the people. In ACL 2010 — Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics. Uppsala, Sweden, 11-16 July 2010, volume 2: Short Papers, pages 68–73.
Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL 2005 — Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics. AnnArbor, Michigan, USA, June 25–30, 2005, pages 115–124.
Richard Socher, Alex Perelygin, Jean Y. Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP 2013 — Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing. Seattle, Washington, USA, 18-21 October 2013, pages 1631–1642.
Carlo Strapparava and Rada Mihalcea. 2007. SemEval-2007 Task 14: Affective text. In SemEval 2007 — Proceedings of the 4th International Workshop on Semantic Evaluations @ ACL 2007. Prague, Czech Republic, June 23-24, 2007, pages 70–74.
