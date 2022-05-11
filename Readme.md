
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

To run `eval_model.py` with `--stel True` you need access to some code from the STEL project and the STEL data (which includes partly proprietary data). You will need to download the STEL repo and ask for access to the STEL data (see: https://github.com/nlpsoc/STEL). 



# Structure

Below you find a structure of the repository and some comments.

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
│   │   └── SD_calc.ipynp 	 # jupyter notebook to calculate stand devation & means from results
│   ├── test
│   │   ├── test_cluster.py 	 	 
│   │   ├── test_eval_model.py 	
│   │   ├── test_fine_tune.py 	
│   │   ├── test_neural_trainer.py 	 	
│   │   └── test_STEL_Or_Content.py 	 
│   └── utility
│       ├── convokit_generator.py 	 	 
│       ├── evaluation_metrics.py 	
│       ├── neural_trainer.py 	
│       ├── plot.py 	 	
│       ├── plot_utility.py 	 	
│       ├── statistics_utility.py 	 	
│       ├── STEL_error_analysis.py 	 	
│       ├── STEL_Or_Content.py 	 	
│       ├── trained_similarities.py 	 	
│       └── plot_utility.py 	 
├── LICENSE
├── Readme.md
└── .gitignore 
```

# Citation

If you find our work or this repository helpful, please consider citing our paper



```
@article{wegmann2022style,
  title={Same Author or Just Same Topic? Towards Content-Independent Style Representations},
  author={Wegmann, Anna and Schraagen, Marijn and Nguyen, Dong},
  journal={arXiv preprint arXiv:2204.04907},
  year={2022}
}
```



# Comments

For comments, problems or questions open an issue or contact Anna (a.m.wegmann@uu.nl). 
