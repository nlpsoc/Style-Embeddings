
Thank you for your interest in Style Embeddings. This is the code that is part of the `Same Author or Just Same Topic? Towards Content-Independent Style Representations` at the 7th RepL4NLP workshop co-located with ACL 2022.

In `Data` you can find the generated training (contrastive) AV tasks with `src/style_embed/generate_dataset.py`.  The best-performing style embedding as trained and described in our publication can be found here: https://huggingface.co/AnnaWegmann/Style-Embedding

# Quickstart

You might just want to use the style embedding model and not fine-tune anything or generate authorship verification tasks. If that is the case it is not necessary to to download anything from the repo. Just use the above [huggingface model](https://huggingface.co/AnnaWegmann/Style-Embedding).

```Python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('AnnaWegmann/Style-Embedding')
embeddings = model.encode(sentences)
print(embeddings)
```



# Prerequisites

To run most functionalities you will only need to have the necessary **Python Packages** installed. To run all evaluations (i.e., including the STEL tasks), you will need to have access to **[STEL](https://github.com/nlpsoc/stel)** data and code.

## Python Packages

Python version **3.8.5**

```
pandas==1.1.3
numpy==1.18.5
pytorch==1.7.1
sentence-transformers==2.1.0
transformers==4.12.2
scipy==1.5.2
unittest==?
```



## STEL

To run `eval_model.py` with `--stel True` you need access to some code from the STEL project and the STEL data (which includes partly proprietary data). You will need to download How the data is accessible can be changed in 



# Structure

When you add all necessary (partly proprietary) data to use ALL  functionalities, the folder should look something like this. Files  starting with _ include proprietary data (see below). They are not  included in the public release but will need to be acquired. The  Datasets folder contains files that were used to generate STEL. They  were not included because of size. Here, the GYAFC_Corpus also needs  permission from Verizon. Everything else in the Datasets folder can be  downloaded from different sources, see also `to_add_const.py`.

```python
.
├── Data
│   └── train_data
│       ├── GENERATE_100-SUBS-2018_745453.txt 
│       ├── Task-Statistics.ipynb 
│       ├── author_data.json
│       ├── dev-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv
│       ├── dev-45000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv
│       ├── dev-45000__subreddits-100-2018_tasks-300000__topic-variable-subreddit.tsv
│       ├── test-45000__subreddits-100-2018_tasks-300000__topic-variable-conversation.tsv
│       ├── test-45000__subreddits-100-2018_tasks-300000__topic-variable-random.tsv
│       ├── test-45000__subreddits-100-2018_tasks-300000__topic-variable-subreddit.tsv
│       ├── train-210000__subreddits-100-2018_tasks-300000__topic-variable-conversation.zip
│       ├── train-210000__subreddits-100-2018_tasks-300000__topic-variable-random.zip
│       └── train-210000__subreddits-100-2018_tasks-300000__topic-variable-subreddit.zip
├── src
│   ├── output
│   ├── style_embed
│   │   ├── cluster.py 	 	 # script to cluster sentences with a rep. model 
│   │   ├── eval_model.py 	 # script to evaluate embeddings on CAV & STEL-Or-Content 
│   │   ├── fine_tune.py 	 # script to fine-tune transformers based on CAV tasks 
│   │   ├── generate_dataset.py 	 # script to generate CAV tasks with different CC variables 
│   │   ├── global_const.py 	 	 # constants & functions that are globally accessible in the project 
│   │   ├── global_identifiable.py 	 # same as const but including paths/names that are local, like dir paths
│   │   └── global_identifiable.py 	 # same as const but including paths/names that are local, like dir paths
│   ├── test
│   │   ├── test_cluster.py 	 	 
│   │   ├── test_eval_model.py 	
│   │   ├── test_fine_tune.py 	
│   │   ├── test_neural_trainer.py 	 	
│   │   └── test_STEL_Or_Content.py 	 
│   └── utility
│       ├── test_cluster.py 	 	 
│       ├── test_eval_model.py 	
│       ├── test_fine_tune.py 	
│       ├── test_neural_trainer.py 	 	
│       └── test_STEL_Or_Content.py 	 
├── LICENSE
├── readme.md
└── .gitignore 
```

To get some statistics about the training tasks use `Task-Statistics.ipynb`

# 

# Citation

# Comments

