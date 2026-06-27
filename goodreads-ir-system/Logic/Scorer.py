import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None
        self._doc_norms = None


    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        query_terms = query.split()
        docs = set()
        
        for term in query_terms:
            if term in self.index:
                docs.update(self.index[term].keys())

        return list(docs)


    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term not in self.index:
            return 0.0
        
        if term not in self.idf:
            df = len(self.index[term])
            # idf = log(N/df)
            self.idf[term] = math.log(self.N / df)

        return self.idf[term]


    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        terms = query.split()
        return dict(Counter(terms))


    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        doc_method_str, query_method_str = method.split('.')

        query_tfs = self.get_query_tfs(query)
        docs = self.get_list_of_documents(query)

        doc_norms = None
        if doc_method_str[2] == 'c':
            doc_norms = self._get_doc_norms(doc_method_str)

        scores = []

        for doc_id in docs:
            score = self.get_vector_space_model_score(
                query, query_tfs, doc_id, doc_method_str, query_method_str
            )
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        # build document vector
        doc_vec = {}
        for term in query.split():
            if term in self.index and document_id in self.index[term]:
                tf_raw = self.index[term][document_id]
                tf = self._apply_tf(tf_raw, document_method[0])
                idf = self.get_idf(term) if document_method[1] == 't' else 1.0
                doc_vec[term] = tf * idf
        
            else:
                doc_vec[term] = 0.0

        # build query vector
        query_vec = {}
        for term, tf_raw in query_tfs.items():
            tf = self._apply_tf(tf_raw, query_method[0])
            idf = self.get_idf(term) if query_method[1] == 't' else 1.0
            query_vec[term] = tf * idf

        if document_method[2] == 'c':
            doc_norm = self._doc_norms[document_id]
            doc_vec = {term: doc_vec[term]/doc_norm for term in query_vec}

        if query_method[2] == 'c':
           query_vec = self._cosine_normalize(query_vec)   

        # dot product for final score
        score = sum(doc_vec.get(term) * query_vec.get(term) for term in query_vec)
        return score


    def compute_scores_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        doc_ids = self.get_list_of_documents(query)
        scores = []

        for doc_id in doc_ids:
            score = self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)
            scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        k1 = 1.2
        b = 0.75
        query_terms = query.split()
        doc_len = document_lengths.get(document_id, 0)
        score = 0.0

        for term in query_terms:
            if term in self.index and document_id in self.index[term]:
                tf_d = self.index[term][document_id]
                idf = self.get_idf(term)
                numerator = (k1 + 1) * tf_d
                denominator = k1 * ((1 - b) + b * (doc_len / average_document_field_length)) + tf_d
                score += idf * numerator / denominator
        return score
    

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        self._prepare_collection_stats()
        doc_ids = self.get_list_of_documents(query)
        scores = []

        for doc_id in doc_ids:
            score = self.compute_score_with_unigram_model(query, doc_id, smoothing_method, document_lengths, alpha, lamda)
            scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        query_terms = query.split()
        doc_len = document_lengths.get(document_id, 0)
        score = 0.0

        for term in query_terms:
            if term in self.index and document_id in self.index[term]:
                tf_d = self.index[term][document_id]
            else:
                tf_d = 0

            cf = self._collection_frequencies.get(term, 0)
            coll_prob = cf / self._collection_length

            if smoothing_method == 'naive':
                prob = tf_d / doc_len

            elif smoothing_method == 'bayes':
                prob = (tf_d + alpha * coll_prob) / (doc_len + alpha)
                
            elif smoothing_method == 'mixture':
                prob = lamda * (tf_d / doc_len) + (1 - lamda) * coll_prob

            else:
                raise ValueError(f"Unknown smoothing method: {smoothing_method}")

            if prob > 0:
                score += math.log(prob)
            else:
                return float('-inf')

        return score


    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n' natural
            - 'l' log

        """
        if mode == 'n':
            return tf
        elif mode == 'l':
            return 1 + math.log(tf) if tf > 0 else 0
        else:
            raise ValueError(f"Unknown tf mode: {mode}")


    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        squared_sum = 0
        for w in weights.values():
            squared_sum += w * w
        norm = math.sqrt(squared_sum)

        if norm == 0:
            return {term: 0.0 for term in weights}
        
        return {term: w / norm for term, w in weights.items()}


    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        if self._collection_frequencies is not None:
            return
        
        collection_frequencies = {}
        collection_length = 0

        for term, posting in self.index.items():
            collection_frequencies[term] = 0
            for _, tf in posting.items():
                collection_frequencies[term] += tf
                collection_length += tf


        self._collection_frequencies = collection_frequencies
        self._collection_length = collection_length

    def _get_doc_norms(self, method):
        """
        Compute and cache the Euclidean norm of the full tf-idf vector for every document.
        Returns {doc_id: norm}
        """
        if self._doc_norms is not None:
            return self._doc_norms

        doc_sum_sq = {}
        for term, postings in self.index.items():
            idf = self.get_idf(term) if method[1] == 't' else 1.0
            for doc_id, tf_raw in postings.items():
                tf = self._apply_tf(tf_raw, method[0])
                weight = tf * idf
                doc_sum_sq[doc_id] = doc_sum_sq.get(doc_id, 0.0) + weight * weight

        self._doc_norms = {doc_id: math.sqrt(sq) for doc_id, sq in doc_sum_sq.items()}
        return self._doc_norms
