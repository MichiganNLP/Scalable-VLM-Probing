# Scalable VLM Probing

[[Paper]](https://arxiv.org/pdf/2305.18786.pdf)

In this work, we propose a simple and effective method to **probe vision-language models**. 

Our method is **scalable**, as it does not require data annotation and makes use of existing datasets. 
With our method, we analyzed the performance of **CLIP**, a popular state-of-the-art multi-modal model, on the **SVO-Probes** benchmark. 

We hope our work contributes to ongoing efforts to discover the limitations of multi-modal models and help build more robust and reliable systems. 
Our framework can be easily used to analyze other benchmarks, features, and multi-modal models

<p style="text-align:center">
    <img src="images/task_overview.png" alt="A description of our probing method, showing 2 images being input to clip, then 3 scores being computed. Different kind of features are used to compute their correlation with each of the scores.">
</p>

## Setup

With Python >= 3.8, run the following commands:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download(['omw-1.4', 'wordnet'])"
spacy download en_core_web_trf
huggingface-cli login
mkdir data
```

**We will post more instructions soon.**

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
    abstract = "Joint vision-language models have shown great performance over a diverse set of tasks. However, little is known about their limitations, as the high dimensional space learned by these models makes it difficult to identify semantic errors. Recent work has addressed this problem by designing highly controlled probing task benchmarks. Our paper introduces a more scalable solution that relies on already annotated benchmarks. Our method consists of extracting a large set of diverse features from a vision-language benchmark and measuring their correlation with the output of the target model. We confirm previous findings that CLIP behaves like a bag of words model and performs better with nouns and verbs; we also uncover novel insights such as CLIP getting confused by concrete words. Our framework is available at this https URL and can be used with other multimodal models and benchmarks.",
}
```
