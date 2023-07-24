import os
import csv

txt_files = []
wav_files = [] 

PATH = 'psychology_book'
# Get list of .txt and .wav files
for f in os.listdir(PATH):
    if f.endswith('.txt'):
        txt_files.append(f)
    elif f.endswith('.wav'):
        wav_files.append(f)
        
with open('metadata.csv', 'w', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['file_name', 'transcription'])
    
    # Write rows
    for txt_file in txt_files:
        wav_file = txt_file.replace('.txt', '.wav')
        
        with open(os.path.join(PATH, txt_file)) as f:
            transcription = f.read().strip()
            
        writer.writerow([wav_file, transcription])