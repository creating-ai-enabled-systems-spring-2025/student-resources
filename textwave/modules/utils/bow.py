import re
from collections import Counter

class Bag_of_Words:
    """
    A Bag-of-Words represnation transformer that learns a vocabulary from a corpus and transforms
    documents into their Bag-of-Words (BoW) representation.

    The BoW model represents text data as a collection of word counts, ignoring the
    order and structure of words. This transformer builds a vocabulary from the provided
    training corpus and then counts occurrences of these vocabulary words in new documents.
    """

    def __init__(self):
        """
        Initializes the Bag_of_Words transformer with an empty vocabulary.

        Attributes:
            vocabulary_ (dict): A dictionary mapping each unique word found in the corpus
                                to a unique index. This is constructed during the fit process.
        """
        self.vocabulary_ = {}

    def _tokenize(self, text):
        """
        Tokenizes the input text by converting it to lowercase and extracting words using a regular expression.

        This basic tokenization approach splits the text on word boundaries, capturing only alphanumeric
        sequences. Adjust the regular expression if you require a different tokenization strategy.

        Parameters:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of word tokens extracted from the text.
        """
        return re.findall(r'\b\w+\b', text.lower())


    def fit(self, documents):
        """
        Learns the vocabulary from the corpus by processing each document and extracting unique tokens.

        During this process, each document in the training corpus is tokenized, and the set of unique
        words is aggregated across all documents. The vocabulary is then created by sorting these unique words
        and assigning each a unique index.

        Parameters:
            documents (list of str): The training corpus where each document is a string.

        Returns:
            Bag_of_Words: The fitted transformer instance with an updated vocabulary_ attribute.
        """
        unique_tokens = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            unique_tokens.update(tokens)

        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}
        return self

    def transform(self, document):
        """
        Transforms a single document into its Bag-of-Words representation.

        This method tokenizes the input document and counts the occurrences of each token that exists
        in the learned vocabulary. The output is a dictionary where keys are tokens (words) and values
        are their corresponding counts in the document.

        Parameters:
            document (str): A single document to be transformed into a BoW vector.

        Returns:
            dict: A dictionary mapping each term (from the learned vocabulary) to its count in the document.
                  Only tokens present in the vocabulary are included.
        """
        tokens = self._tokenize(document)
        token_counts = Counter(tokens)

        bow_vector = {
            token: count for token, count in token_counts.items()
            if token in self.vocabulary_
        }
        return bow_vector


if __name__ == "__main__":
    # Example corpus of 9 documents to train the Bag-of-Words representation.
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

    # Fit the transform on the corpus.
    transform = Bag_of_Words()
    transform.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    bow_test = transform.transform(test_document)
    
    print("Test Document Bag-of-Words:")
    for term, count in sorted(bow_test.items()):
        print(f"  {term}: {count}")
