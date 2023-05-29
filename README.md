# Scalable VLM Probing

Probing CLIP on SVO-Probes.

## Overview

![Example instance](images/task_description.jpg)

## Setup

With Python >= 3.8, run the following commands:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download(['omw-1.4', 'wordnet'])"
spacy download en_core_web_trf
huggingface-cli login
```
