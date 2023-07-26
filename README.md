# ☀️ Whisper Kurdish

This repo contains code to finetune the Whisper model on Central Kurdish speech data. 

## Contents

- `train_tokenizer.py`: Trains a Central Kurdish tokenizer on text from the OSCAR corpus
- `dataset.py`: Combines multiple Central Kurdish speech datasets, vectorizes them using the tokenizer, and pushes to the 🤗 Hub  
- `utils/`: Scripts to format datasets like Common Voice and Asosoft speech into 🤗 dataset format
- `finetune.py`: Finetunes Whisper on the combined dataset and pushes to 🤗 Hub

## Usage