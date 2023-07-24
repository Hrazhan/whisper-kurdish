from datasets import load_dataset
from transformers import WhisperTokenizerFast
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="whisper-kurdish", type=str, help="Repo id the tokenizer to be pushed to")
parser.add_argument("--push_to_hub", action="store_true", help="Push to hub",)

args = parser.parse_args()


dataset = load_dataset("oscar-corpus/OSCAR-2301", "ckb", split="train", use_auth_token=True)

def get_training_corpus(batch_size=1000):
    for start_idx in range(0, len(dataset), batch_size):
        samples = dataset[start_idx : start_idx + batch_size]
        yield samples["text"]

training_corpus = get_training_corpus()

tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-small")

tokenizer = tokenizer.train_new_from_iterator(
    training_corpus, vocab_size=50257,
)

processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
processor.tokenizer = tokenizer
processor.save_pretrained(args.model_name, push_to_hub=args.push_to_hub)

