import pandas as pd

def is_complete_sentence(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return False
    valid_endings = ('.', '!', '?', '"', '”', "'")
    return text.strip().endswith(valid_endings)

for file in ['data/processed/summary_data.csv', 'data/processed/summary_data1.csv']:
    try:
        df = pd.read_csv(file)
        print(f'\n--- Analyzing {file} ---')
        print(f'Total rows: {len(df)}')
        if 'summary' not in df.columns:
            print('No summary column!')
            continue
        
        df['summary_len'] = df['summary'].astype(str).apply(lambda x: len(x.split()))
        df['text_len'] = df['text'].astype(str).apply(lambda x: len(x.split()))
        
        truncated_summaries = df['summary'].astype(str).apply(lambda x: not is_complete_sentence(x)).sum()
        short_summaries = (df['summary_len'] < 15).sum()
        
        print(f'Avg text length: {df["text_len"].mean():.2f} words')
        print(f'Avg summary length: {df["summary_len"].mean():.2f} words')
        print(f'Truncated summaries (no end punctuation): {truncated_summaries}')
        print(f'Short summaries (< 15 words): {short_summaries}')
        
        print('\nSample summary (row 0):', df['summary'].iloc[0])
    except Exception as e:
        print(f'Error reading {file}: {e}')
