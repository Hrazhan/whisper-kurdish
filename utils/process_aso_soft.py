import os
import zipfile
import pandas as pd

input_dir = 'AsoSoft-Speech-Testset' 
output_dir = 'clips'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame(columns=['file_name', 'transcription']) 

for filename in os.listdir(input_dir):
    if filename.endswith('.zip'):
        fullpath = os.path.join(input_dir, filename)
        with zipfile.ZipFile(fullpath, 'r') as zip_ref:
            zip_ref.extractall(input_dir)
            foldername = os.path.splitext(filename)[0]
            subpath = os.path.join(input_dir, foldername)
            
            rows = []
            for wav_name in os.listdir(subpath):
                if wav_name.endswith('.wav'):
                    wav_path = os.path.join(subpath, wav_name)
                    wrd_name = os.path.splitext(wav_name)[0] + '.wrd'
                    wrd_path = os.path.join(subpath, wrd_name)
                    
                    with open(wrd_path, 'r') as f:
                        transcription = f.read().strip()
                        
                    row = {'file_name': wav_name, 'transcription': transcription}
                    rows.append(row)
                    
                    output_wav = os.path.join(output_dir, wav_name)
                    os.rename(wav_path, output_wav)
                    
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) 

df.to_csv('transcriptions.csv', index=False)