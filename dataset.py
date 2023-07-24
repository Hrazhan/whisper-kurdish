from datasets import Audio, DatasetDict,  load_dataset, concatenate_datasets, Dataset
from transformers import WhisperProcessor, AutoTokenizer
from klpt.preprocess import Preprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="whisper-kurdish", type=str, help="Repo id to get the tokenizer & feature extraction from")
parser.add_argument("--dataset_name", default="deng", type=str, help="Dataset repo id")
parser.add_argument("--sampling_rate", default=16000, type=int, help="Sampling rate")
parser.add_argument("--num_proc", default=1, type=int, help="Number of processes to use")

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


preprocessor_ckb = Preprocess("Sorani", "Arabic", numeral="Latin")



processor = WhisperProcessor.from_pretrained(args.model_name, task="transcribe")


def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["transcription"]
    transcription = preprocessor_ckb.preprocess(transcription)
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids

    return batch



deng = raw_datasets.map(prepare_dataset, remove_columns=list(next(iter(raw_datasets.values())).features), num_proc=args.num_proc).with_format("torch")
deng.push_to_hub(args.dataset_name, max_shard_size="2000MB")

