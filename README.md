# NeuralCoref

## Overview

* Pronoun resolution is the task of linking together nouns and pronouns correctly, and is essential for language understanding.
* When datasets are biased, trained models amplify these biases
* If biased models influence decisions, we risk discrimination based on gender


## Abstract
In this paper we study gender bias in neural pronoun resolution models. Pronoun resolutions is the task of linking a pronoun to the correct noun. We find that pre-trained state-of-the-art neural models tend to be biased. Their underlying bias is caused by the data these models are trained on. We show that one of most commonly used dataset for training coreference resolution models has a substantial bias towards male entities, which causes models to perform better on male examples. Since this can lead to discrimination for e.g. job applicants, we are motivated to highlight the problem and explore methods to mitigate this gender bias. We find that using gender neutral word embeddings such as the Debiased GoogleNews embeddings or the Word Dependency embeddings can lower the bias considerably in the model predictions. Furthermore, we show through experimental work how making all pronouns gender neutral can mitigate gender bias. We believe that these two approaches can help avoid gender stereotypical decision-making. We test these bias reducing techniques on state-of-the-art model presented by together with a modification of this model minded to the given task presented in, which is a balanced dataset with equal amounts of male and female examples. We find these methods have the ability of lowering gender bias. 

## Data

In this project, we worked with the Gendered Ambiguous Pronouns (GAP) dataset. Below is an example of how this dataset is formatted.

![alt text](/images/example.png)

We compared the gender bias in this dataset with the commonly used OntoNotes dataset.

![alt text](/images/onto_notes.png)

![alt text](/images/gap.png)

## Our toolbox for mitigating gender bias

### Use gender neutral pronouns

Here is examples of different gender neutral pronoun schemes. We experimented with all of them.

![alt text](/images/pronoun_scheme.png)

### Unbiased embeddings

We found that most word embeddings have gender bias. This is probably inheriented from the data they have been trained on. Here we show a comparison between the commonly used Glove 840 300 word embedding and the Google News Debiased embedding. We check for gender bias as the distance between a job title and the word man / woman. We use 20 stereotypical female / male jobs (red / blue in the plots). More precisely, we calculate

$$d_i = ||job_i - man|| - ||job_i - woman||$$

This means that a negative distance indicates that the job title is more closer to a man than woman and vice versa. We see that the Google Debiased Embedding has significantly less bias than the Glove Embedding.

Glove Embedding
![alt text](/images/glove.png)

Google News Debiased Embedding
![alt text](/images/google_debiased.png)

We take the absolute distance of all the distance calculate by the above discribed approach, such that we can compare multiple embeddings for gender bias in one plot.

![alt text](/images/comparison.png)

We see that the Word Dependency Embedding and the Google News Debiased Embedding has the lowest gender bias.

## Results

We report the gender bias and the performance for our tested models in the following to tables.

![alt text](/images/performance.png)

![alt text](/images/bias.png)







