import os
import pandas as pd
import shutil

train_files =  ['validated.tsv', 'dev.tsv', 'train.tsv', 'invalidated.tsv', 'other.tsv'] 
# train_files = [file for file in os.listdir('./') if file.endswith('.tsv') and file != 'test.tsv']
test_file = 'test.tsv'

train_dir = 'train'
test_dir = 'test'

file_dir = "clips"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_df = pd.DataFrame(columns=['file_name', 'transcription'])
test_df = pd.DataFrame(columns=['file_name', 'transcription'])

for file in train_files:
    df = pd.read_csv(file, sep='\t')
    df = df.rename(columns={'path': 'file_name', 'sentence': 'transcription'})

    for file_name in df['file_name']:
        src = os.path.join( file_dir, file_name)
        dst = os.path.join(train_dir, file_name)
        shutil.copy(src, dst)

    train_df = pd.concat([train_df, df])


test_df = pd.read_csv(test_file, sep='\t')
test_df = test_df.rename(columns={'path': 'file_name', 'sentence': 'transcription'})


for file_name in test_df['file_name']:
    src = os.path.join(file_dir, file_name)
    dst = os.path.join(test_dir, file_name)
    shutil.copy(src, dst)

test_df = test_df[['file_name', 'transcription']]
train_df = train_df[['file_name', 'transcription']]

train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# Find duplicate file names between dataframes
test_names = test_df['file_name'].tolist()

# Find duplicates in train_df
duplicates = train_df[train_df['file_name'].isin(test_names)]


# Drop the test set from train_df
train_df = train_df.loc[~train_df["file_name"].isin(test_names)]

# Get list of duplicate file paths
duplicate_paths = [os.path.join(train_dir, name) for name in duplicates['file_name']]


# Delete duplicate files from train folder
for path in duplicate_paths:
    os.remove(path)


train_df.to_csv('train.csv', index=False)  
test_df.to_csv('test.csv', index=False)
