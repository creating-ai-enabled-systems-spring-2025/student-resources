import math
import re
from collections import Counter, defaultdict

class TF_IDF:
    """
    A TF-IDF transformer that learns a vocabulary from a corpus and transforms
    documents into their TF-IDF representation.
    
    The TF-IDF (Term Frequency - Inverse Document Frequency) model is a numerical
    statistic intended to reflect how important a word is to a document in a corpus.
    This transformer builds a vocabulary of unique tokens from the provided corpus and
    computes an IDF score for each token. New documents can then be transformed into
    a dictionary where each token is associated with its TF-IDF score.
    
    Attributes:
        vocabulary_ (dict): A mapping of tokens (str) to unique indices (int), created
                            during the fitting process.
        idf_ (dict): A mapping of tokens to their computed inverse document frequency values.
    """

    def __init__(self):
        """
        Initializes the TF_IDF transformer with an empty vocabulary and IDF mapping.
        
        This constructor sets up the transformer without any pre-loaded vocabulary.
        The vocabulary and IDF values will be computed when the 'fit' method is called.
        """
        self.vocabulary_ = {}
        self.idf_ = {}

    def _tokenize(self, text):
        """
        Tokenizes the input text by converting it to lowercase and extracting words.
        
        The function uses a regular expression to match word boundaries and extract
        alphanumeric sequences as tokens. This is a basic tokenization approach that
        may be extended for more complex use cases.
        
        Parameters:
            text (str): The text to tokenize.
            
        Returns:
            list: A list of word tokens (str) extracted from the input text.
            
        Example:
            >>> tokens = TF_IDF()._tokenize("Hello World!")
            >>> print(tokens)
            ['hello', 'world']
        """
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, documents):
        """
        Learns the vocabulary and computes the inverse document frequency (IDF) from the corpus.
        
        The 'fit' method processes each document in the provided corpus, tokenizes them,
        and constructs a set of unique tokens. It then calculates the document frequency
        for each token (i.e., the number of documents that contain the token). The IDF for
        each token is computed using the formula:
        
            IDF(token) = log(total_documents / (document_frequency + 1)) + 1
        
        The vocabulary is stored as a mapping from token to index, and the IDF values
        are stored in a separate dictionary.
        
        Parameters:
            documents (list of str): A list of documents (each document is a string)
                                     that forms the training corpus.
                                     
        Returns:
            TF_IDF: The instance of the TF_IDF transformer with the learned vocabulary and IDF values.
            
        Example:
            >>> corpus = ["The quick brown fox.", "Lazy dog."]
            >>> transformer = TF_IDF().fit(corpus)
            >>> print(transformer.vocabulary_)
            {'brown': 0, 'dog': 1, 'fox': 2, 'lazy': 3, 'quick': 4, 'the': 5}
        """
        doc_freq = defaultdict(int)
        total_docs = len(documents)

        for doc in documents:
            tokens = set(self._tokenize(doc))  # Unique tokens in this doc
            for token in tokens:
                doc_freq[token] += 1

        # Compute IDF
        self.idf_ = {
            token: math.log(total_docs / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }

        # Create vocabulary mapping each token to an index
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(doc_freq))}
        return self

    def transform(self, document):
        """
        Transforms a document into its TF-IDF representation.
        
        This method tokenizes the input document and computes the term frequency (TF) for each token.
        The TF is normalized by dividing the token count by the total number of tokens in the document.
        Each token's TF value is then multiplied by its corresponding IDF value (learned during 'fit')
        to obtain the TF-IDF score. Only tokens present in the learned vocabulary are included.
        
        Parameters:
            document (str): A single document (string) to be transformed.
            
        Returns:
            dict: A dictionary where keys are tokens (str) and values are the corresponding TF-IDF scores (float).
            
        Example:
            >>> transformer = TF_IDF().fit(["The quick brown fox.", "Lazy dog."])
            >>> tfidf_vector = transformer.transform("The quick fox.")
            >>> print(tfidf_vector)
            {'fox': 0.709, 'quick': 0.709, 'the': 0.354}
        """
        tokens = self._tokenize(document)
        total_tokens = len(tokens)
        token_counts = Counter(tokens)

        tfidf_vector = {}
        for token, count in token_counts.items():
            if token in self.idf_:
                tf = count / total_tokens
                tfidf = tf * self.idf_[token]
                tfidf_vector[token] = tfidf

        return tfidf_vector


if __name__ == "__main__":
    # Example corpus of 9 documents to train the TF-IDF transformer.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    # Fit the transformer on the corpus.
    transformer = TF_IDF()
    transformer.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)
    
    # Display the TF-IDF representation of the test document.
    print("Test Document TF-IDF:")
    for term, score in sorted(tfidf_test.items()):
        print(f"  {term}: {score:.4f}")
