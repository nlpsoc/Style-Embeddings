
Thank you for your interest in Style Embeddings. This is the code that is part of the [Same Author or Just Same Topic? Towards Content-Independent Style Representations](https://aclanthology.org/2022.repl4nlp-1.26/) publication at the 7th RepL4NLP workshop co-located with ACL 2022.

In `Data` you can find the generated training (contrastive) AV tasks with `src/style_embed/generate_dataset.py`.  The best-performing style embedding as trained and described in our publication can be found here: https://huggingface.co/AnnaWegmann/Style-Embedding

# Quickstart

You might just want to use the style embedding model and not fine-tune anything or generate authorship verification tasks. If that is the case it is not necessary to to download anything from the repo. Just use the above [huggingface model](https://huggingface.co/AnnaWegmann/Style-Embedding). The Huggingface Hosted Inference API also allows calculating sentence similarities without downloading anything if you want to just try out a few sentence similarities.

To load the model from the huggingface hub and encode a sentence:
```Python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('AnnaWegmann/Style-Embedding')
embeddings = model.encode(sentences)
print(embeddings)
```

```
[[ 1.4103483e-01  2.0874986e-01  1.7136575e-01 ...  3.5445389e-01
   4.6482438e-01 -4.4582412e-06]
 [ 3.9674792e-01  1.5319356e-01  7.7177114e-03 ...  6.6641623e-01
   4.8512089e-01 -3.2561386e-01]]
```

Let's calculate the sentence similarity between two sentences with our new style model. We use a prallel example with formal vs. informal style from [GYAFC](https://aclanthology.org/N18-1012/). See also Figure 2 in our paper. 
```Python
from sentence_transformers import util

emb1 = model.encode("r u a fan of them or something?")  # more informal sentence
emb2 = model.encode("Are you one of their fans?")  # more formal sentence with similar content to emb1
print("Cosine-Similarity:", util.cos_sim(emb1, emb2))
```

```
Cosine-Similarity: tensor([[0.078]])
```

```Python
emb3 = model.encode("Oh yea and that young dr got a bad haircut")  # more informal sentence with different content from emb1
print("Cosine-Similarity:", util.cos_sim(emb1, emb3))
```

```
Cosine-Similarity: tensor([[0.745]])
```


## Fine-tuning 

If you want to fine-tune a RoBERTa model based on the CAV task, you will need the provided code. 

First step: generate the training task based on [convokit](https://convokit.cornell.edu/). If you want to use your own conversation data, you will need to first load it in convokit corpus format. For example, [via a pandas dataframe](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/corpus_from_pandas.ipynb).

```python
from utility.convokit_generator import TaskGenerator
from global_const import TOPIC_CONVERSATION

# these can take some time depending on your data size ...
cav_gen = TaskGenerator(convokit_data_keys=["subreddit-ApplyingToCollege"], years=[2018], total=10)
cav_gen._get_data_split(topic_variable=TOPIC_CONVERSATION)
train_path, dev_path, test_path = cav_gen.save_data_split(output_dir=".", topic_variable=TOPIC_CONVERSATION)
```

Second step: use the generated data as a fine-tuning task.

```python
from utility.neural_trainer import SentenceBertFineTuner
from utility.training_const import TRIPLET_LOSS, TRIPLET_EVALUATOR

tuner = SentenceBertFineTuner(model_path="roberta-base", train_filename=train_path, dev_filename=dev_path, loss=TRIPLET_LOSS, evaluation_type=TRIPLET_EVALUATOR)
model_path = tuner.train(epochs=1, batch_size=8)

from sentence_transformers import util
emb1 = tuner.model.encode("I'm a happy person")
emb4 = tuner.model.encode("I'm a happy person.")
print("Cosine-Similarity:", util.cos_sim(emb1, emb4))
```

Third step: evaluating the trained model (without the STEL directory)

```python
from eval_model import _evaluate_model
from utility.trained_similarities import TunedSentenceBertSimilarity
from global_const import set_logging
set_logging()

model_w_sim = TunedSentenceBertSimilarity(model=tuner.model)
_evaluate_model(model_w_sim, model_path, [test_path], test_stel=False, test_AV=True)
```

Evaluating WITH STEL (needs the downloaded repo on the same level as style-embeddings)

```python
from global_identifiable import include_STEL_project 
include_STEL_project()

_evaluate_model(model_w_sim, model_path, [test_path], test_stel=True, test_AV=False)
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
@inproceedings{wegmann-etal-2022-author,
    title = "Same Author or Just Same Topic? Towards Content-Independent Style Representations",
    author = "Wegmann, Anna  and
      Schraagen, Marijn  and
      Nguyen, Dong",
    booktitle = "Proceedings of the 7th Workshop on Representation Learning for NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.repl4nlp-1.26",
    pages = "249--268",
    abstract = "Linguistic style is an integral component of language. Recent advances in the development of style representations have increasingly used training objectives from authorship verification (AV){''}:'' Do two texts have the same author? The assumption underlying the AV training task (same author approximates same writing style) enables self-supervised and, thus, extensive training. However, a good performance on the AV task does not ensure good {``}general-purpose{''} style representations. For example, as the same author might typically write about certain topics, representations trained on AV might also encode content information instead of style alone. We introduce a variation of the AV training task that controls for content using conversation or domain labels. We evaluate whether known style dimensions are represented and preferred over content information through an original variation to the recently proposed STEL framework. We find that representations trained by controlling for conversation are better than representations trained with domain or no content control at representing style independent from content.",
}
```



# Comments

For comments, problems or questions open an issue or contact Anna (a.m.wegmann@uu.nl). 
