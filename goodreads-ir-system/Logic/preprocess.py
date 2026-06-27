import re
import string
import json
import csv
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self, custom_stopwords_path='./stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_pattern = re.compile(pattern)

        self.stopwords = set()
        try:
            with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(f.read().splitlines())
        except FileNotFoundError:
            print("Error loading stopwords!")

        self.stemmer = PorterStemmer()


    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        # 1. Strip URLs and emails using RegEx.
        text = self.url_pattern.sub('', text)

        # 2. Case-folding and punctuation removal.
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 3. Filter out non-informative words using `stopwords.txt`.
        tokens = self.remove_stopwords(text)

        # 4. Apply the Stemmer or Lemmatizer to reduce words to their root forms (e.g., "running" → "run").
        tokens = [self.normalize(t) for t in tokens]

        return ' '.join(tokens)

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        tokens = text.split()
        return [word for word in tokens if word not in self.stopwords]
        
    
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        list
            The normalized word.
        """
        return self.stemmer.stem(word)

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        return [self.preprocess_text(doc) for doc in documents]
    


def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()

    for doc in docs:
        # title
        title = doc.get("title", "")
        if isinstance(title, list):
            title = " ".join(title)
        doc["title"] = preprocessor.preprocess_text(title)

        # description
        desc = doc.get("description", "")
        if isinstance(desc, list):
            desc = " ".join(desc)
        doc["description"] = preprocessor.preprocess_text(desc)

        # author
        author = doc.get("author", "")
        if isinstance(author, list):
            author = " ".join(author)
        doc["author"] = preprocessor.preprocess_text(author)


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    books = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map the fields
                book = {
                    "id": row.get("bookId", "").strip(),
                    "title": row.get("title", "").strip(),
                    "author": row.get("author", "").strip(),
                    "description": row.get("description", "").strip(),
                    # Split comma-separated fields, strip whitespace, filter empty strings
                    "genres": [g.strip() for g in row.get("genres", "").split(",") if g.strip()],
                    "characters": [c.strip() for c in row.get("characters", "").split(",") if c.strip()],
                    "languages": [l.strip() for l in row.get("languages", "").split(",") if l.strip()],
                    "publish_date": row.get("publish_date", "").strip(),
                    "num_pages": int(row["num_pages"]) if row.get("num_pages") and row["num_pages"].strip().isdigit() else 0,
                    "avg_rating": float(row["avg_rating"]) if row.get("avg_rating") and row["avg_rating"].strip() else 0.0
                }
                books.append(book)
    except FileNotFoundError:
        print("File not found!")
        return

    # Write JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(books, f, indent=2, ensure_ascii=False)
    print(f"Successfully converted {len(books)} books to '{json_file_path}'.")


if __name__ == '__main__':
    csv_to_json('top_3000_rated_books.csv', 'crawled.json')
    
    with open('crawled.json', 'r', encoding='utf-8') as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open('preprocessed.json', 'w', encoding='utf-8') as file:
        json.dump(docs, file, indent=2, ensure_ascii=False)

    print("Successfully preprocessed the fields (title, description, author) and saved to 'preprocessed.json'.")
