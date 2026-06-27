import string
from collections import deque
from typing import Callable, List, Tuple, Dict

class Snippet:
    """
    A class to generate relevant text snippets from documents based on a search query.
    
    It uses a single-pass sliding window with a 'lag' mechanism to ensure keywords
    at the very beginning and very end of documents are treated as potential centers.
    """

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable, number_of_words_on_each_side: int = 5):
        """
        Initialize the Snippet generator.

        Args:
            normalize_function (Callable): A function that takes a word and returns its stemmed/normalized version.
            remove_stopword_function (Callable): A function that takes a query string and returns a list of filtered tokens.
            number_of_words_on_each_side (int): Number of words to include to the left and right of a keyword.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = number_of_words_on_each_side + 1


    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Main orchestrator for snippet generation.

        Parameters:
            raw_doc (str): The original document string.
            query (str): The user's search query string.

        Returns:
            final_snippet (str): The formatted snippet with '***' highlighting and '...' separators.
            not_exist_words (list): The list of words from the query that were not found in the document.
        """
        if isinstance(raw_doc, list):
            raw_doc = " ".join(raw_doc)

        doc_tokens = raw_doc.split()

        normalized_cache = []
        for token in doc_tokens:
            clean = token.strip(string.punctuation).lower()
            normalized_cache.append(self.normalize(clean))

        query_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
        query_tokens = self.remove_stopword(query_clean)          # list of non‑stopword tokens
        query_stems = [self.normalize(tok) for tok in query_tokens]
        query_set = set(query_stems)

        # determine missing query terms
        doc_stems_set = set(normalized_cache)
        not_exist_words = []
        for stem, token in zip(query_stems, query_tokens):
            if stem not in doc_stems_set:
                not_exist_words.append(token)

        # find best windows
        windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        if not windows:
            return "", not_exist_words

        # merge overlapping/adjacent windows
        merged = self._merge_windows(windows)

        # build highlighted snippet string
        snippet_text = self._create_snippet_text(doc_tokens, normalized_cache, merged, query_set)
        return snippet_text, not_exist_words


    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        """
        Uses a sliding window to score the 'density' of query matches.
        
        Parameters:
            doc_tokens (list): List of original words from the document.
            normalized_cache (list): List of the same words, but normalized/stemmed.
            query_set (set): Set of normalized query stems.

        Returns:
            list: A list of (start_index, end_index) for the best windows found.
        """
        n = len(doc_tokens)
        # special case
        if n < self.win_size:
            count = sum(1 for stem in normalized_cache if stem in query_set)
            if count > 0:
                return [(0, n-1)]
            return []

        best_windows = []
        max_count = 0

        for i in range(n - self.win_size + 1):
            window = normalized_cache[i:i+self.win_size]
            count = sum(1 for stem in window if stem in query_set)
            if count > max_count:
                best_windows = [(i, i+self.win_size-1)]
                max_count = count
            elif count == max_count and max_count > 0:
                best_windows.append((i, i+self.win_size-1))

        return best_windows


    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Combines window ranges that overlap or touch.
        
        Parameters:
            windows (list): List of (start, end) index tuples.

        Returns:
            list: List of merged (start, end) index tuples.
        """
        if not windows:
            return []
            
        windows.sort(key=lambda w: w[0])
        merged = [windows[0]]

        for current_start, current_end in windows[1:]:
            previous_start, previous_end = merged[-1]
            if current_start <= previous_end + 1:
                merged[-1] = (previous_start, max(previous_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged


    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list, 
                             merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        """
        Constructs the final formatted snippet string.
        
        Parameters:
            doc_tokens (list): Original document tokens.
            normalized_cache (list): Stemmed document tokens.
            merged_windows (list): Merged (start, end) indices.
            query_set (set): Normalized query stems.

        Returns:
            str: The final snippet with highlights and ellipses. 
                example: "The ***wizard*** went to ***Hogwarts.*** The ***wizard*** loved magic."

        """
        parts = []
        for start, end in merged_windows:
            window_tokens = []
            for i in range(start, end+1):
                token = doc_tokens[i]
                if normalized_cache[i] in query_set:
                    token = f"***{token}***"
                window_tokens.append(token)
            parts.append(' '.join(window_tokens))

        return ' ... '.join(parts)
