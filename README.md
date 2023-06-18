# Scalable VLM Probing

[[Paper]](https://arxiv.org/pdf/2305.18786.pdf)

In this work, we propose a simple and effective method to **probe vision-language models** (VLMs). 

Our method is **scalable**, as it does not require data annotation and makes use of existing datasets. 
With our method, we analyzed the performance of [CLIP](https://openai.com/research/clip), a popular state-of-the-art
multi-modal model, on the [SVO-Probes](https://github.com/deepmind/svo_probes) benchmark. 

![A description of our probing method, showing 2 images being input to CLIP, then 3 scores being computed. Different
kind of features are used to compute their correlation with each of the scores.](images/task_overview.png)

We hope our work contributes to ongoing efforts to discover the limitations of multi-modal models and help build more
robust and reliable systems. Our framework can be easily used to analyze other benchmarks, features, and multi-modal
models.

## Obtained Results

Under [results/](results) you can find the detailed results obtained with our method for the 3 different scores tested
(read the paper for details). They come from the output of running the code in this repository (see below to reproduce
it).

## Reproducing the Results

### Setup

With Python >= 3.8, run the following commands:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download(['omw-1.4', 'wordnet'])"
spacy download en_core_web_trf
huggingface-cli login
mkdir data
```

**We'll write more instructions soon.**

## Citation

```bibtex
@inproceedings{castro-etal-2023-scalable,
    title = "Scalable Performance Analysis for Vision-Language Models",
    author = "Castro, Santiago  and
      Ignat, Oana  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 12th Joint Conference on Lexical and Computational Semantics",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    abstract = "Joint vision-language models have shown great performance over a diverse set of
    tasks. However, little is known about their limitations, as the high dimensional space learned
    by these models makes it difficult to identify semantic errors. Recent work has addressed this
    problem by designing highly controlled probing task benchmarks. Our paper introduces a more
    scalable solution that relies on already annotated benchmarks. Our method consists of
    extracting a large set of diverse features from a vision-language benchmark and measuring
    their correlation with the output of the target model. We confirm previous findings that CLIP
    behaves like a bag of words model and performs better with nouns and verbs; we also uncover
    novel insights such as CLIP getting confused by concrete words. Our framework is available
    at https://github.com/MichiganNLP/Scalable-VLM-Probing a and can be used with other
    multimodal models and benchmarks.",
}
```
