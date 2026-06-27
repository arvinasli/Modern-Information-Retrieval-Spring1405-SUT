import pickle
import os
import string


class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}


    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        word = f"${word}$"
        return {word[i:i+k] for i in range(len(word) - k + 1)}


    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        if len(first_set | second_set) == 0:
            return 0.0
        
        return len(first_set & second_set) / len(first_set | second_set)


    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_k_gram_words = {}
        word_counter = {}

        for doc in all_documents:
            words = doc.split()
            for word in words:
                word = word.strip(string.punctuation).lower()  # fixes bugs with spell correction suggestions
                if word not in word_counter:
                    word_counter[word] = 0
                    all_k_gram_words[word] = self.k_gram_word(word)
                word_counter[word] += 1
        
        return all_k_gram_words, word_counter


    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)


    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']


    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        word = word.strip(string.punctuation).lower()
        # if word is already in the vocabulary
        if word in self.all_k_gram_words:
            return [word]
        
        first_grams = self.k_gram_word(word)
        candidates = []
        for vocab_word, vocab_grams in self.all_k_gram_words.items():
            jaccard_sc = self.jaccard_score(first_grams, vocab_grams)
            candidates.append((jaccard_sc, vocab_word))

        candidates.sort(key= lambda x: x[0], reverse=True)
        top5 = candidates[:5]

        # scoring by frequency: jaccard score * tf
        scored = []
        for jaccard_sc, word in top5:
            tf = self.word_counter[word]
            score = jaccard_sc * tf
            scored.append((score, word))

        scored.sort(key= lambda x: x[0], reverse=True)
        return [word for _, word in scored]


    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        tokens = query.split()
        corrected = []

        for token in tokens:
            best_candidate = self.find_nearest_words(token)[0]
            corrected.append(best_candidate)

        return ' '.join(corrected)
    
if __name__ == '__main__':
    # Fake corpus
    docs = ["this is a test document with some words", "another document testing the spell checker"]
    sc = SpellCorrection(all_documents=docs, save_path='spell_data.pkl')
    print(sc.spell_check("tets docment"))