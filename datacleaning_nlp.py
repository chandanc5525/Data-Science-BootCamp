import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()  # 'str' accessor not needed for individual string    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)  
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)   
    # Remove mentions
    text = re.sub(r'@\S+', '', text)  
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text) 
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

data['cleaned_text'] = data['reviewText'].apply(clean_text)
