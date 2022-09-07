# CLIP_SVO-Probes

Probing CLIP on SVO-Probes.

## Setup:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('omw-1.4')"
```

## Open a console on Great Lakes

```bash
srun --partition=standard --account=mihalcea0 --cpus-per-task=36 --mem=180G --pty bash
```
