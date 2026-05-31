import re

def clean_text(text: str)->str:
    abstract = text

    # Remove \n characters in the middle and leading/trailing spaces
    abstract = abstract.strip().replace('\n', ' ')

    # Remove special characters
    abstract = re.sub(r'[^\w\s]', '', abstract)

    # Remove digits
    abstract = re.sub(r'\d+', '', abstract)

    # Remove extra spaces
    abstract = re.sub(r'\s+', ' ', abstract).strip()

    # Convert to lowercase
    abstract = abstract.lower()

    return abstract

def preprocess_samples(samples)->list:
    preprocessed_samples = []

    for s in samples:
        abstract = clean_text(s['abstract'])

        # for the label, we only keep the first part
        parts = s['categories'].split(' ')
        category = parts[0].split('.')[0]
    
        preprocessed_samples.append({
            'text': abstract,
            'label': category
        })

    return preprocessed_samples