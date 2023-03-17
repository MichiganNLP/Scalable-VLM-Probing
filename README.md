# CLIP_SVO-Probes

Probing CLIP on SVO-Probes.

## Setup

First, download the data from
[the Google Drive folder](https://drive.google.com/drive/folders/1Cs3sBmYzvXtTwPeDmkGgegdHqGTM08Ov?usp=sharing) into
`data/`.

Then, with Python >= 3.8, run the following commands:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download(['omw-1.4', 'wordnet'])"
huggingface-cli login
```

## Open a console on Great Lakes

```bash
salloc --partition=standard --account=mihalcea98 --cpus-per-task=36 --mem=180G
```
