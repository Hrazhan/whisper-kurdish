from datasets import Audio, DatasetDict,  load_dataset, concatenate_datasets, Dataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="whisper-kurdish", type=str, help="Repo id to get the tokenizer & feature extraction from")
parser.add_argument("--dataset_name", default="deng", type=str, help="Dataset repo id")
parser.add_argument("--sampling_rate", default=16000, type=int, help="Sampling rate")

args = parser.parse_args()

sampling_rate = args.sampling_rate

cv = load_dataset("audiofolder", data_dir="common_voice_14/", split="train").cast_column("audio", Audio(sampling_rate))
aso = load_dataset("audiofolder", data_dir="asosoft-speech", split="train").cast_column("audio", Audio(sampling_rate))
gf = load_dataset("google/fleurs", "ckb_iq", split="train+validation", use_auth_token=True).cast_column("audio", Audio(sampling_rate))
book = load_dataset("audiofolder", data_dir="psychology_book/", split="train").cast_column("audio", Audio(sampling_rate))


gf = gf.remove_columns(["transcription"])
gf = gf.rename_column("raw_transcription", "transcription")
gf = gf.remove_columns(set(gf.features.keys()) - set(["audio", "transcription"]))

all_datasets = [cv, aso, gf, book]

new_dataset = concatenate_datasets(all_datasets)


raw_datasets = DatasetDict()

raw_datasets["train"] = new_dataset
raw_datasets["test"] = load_dataset("audiofolder", data_dir="common_voice_14/", split="test").cast_column("audio", Audio(sampling_rate))


raw_datasets.push_to_hub(args.dataset_name, num_shards={'train': 5, 'test': 1})

