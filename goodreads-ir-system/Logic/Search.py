from Logic.preprocess import Preprocessor
from Logic.Scorer import Scorer
from Logic.indexer import Indexes, Index_types, Index_reader


class SearchEngine:
    def __init__(self, path="indexes/"):
        """
        Initializes the search engine based on your indexing structure.
        """
        self.path = path
        self.fields = [Indexes.CHARACTERS, Indexes.GENRES, Indexes.DESCRIPTIONS]

        self.document_indexes = {
            Indexes.CHARACTERS: Index_reader(path, Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(path, Indexes.DESCRIPTIONS).index,
        }

        self.tiered_index = {}
        for field in self.fields:
            try:
                self.tiered_index[field] = Index_reader(path, field, Index_types.TIERED).index
            except Exception:
                self.tiered_index[field] = {
                    "first_tier": self.document_indexes[field],
                    "second_tier": {},
                    "third_tier": {},
                }

        self.document_lengths_index = {
            Indexes.CHARACTERS: Index_reader(path, Indexes.CHARACTERS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.DESCRIPTIONS: Index_reader(path, Indexes.DESCRIPTIONS, Index_types.DOCUMENT_LENGTH).index,
        }

        self.documents_index = Index_reader(path, Indexes.DOCUMENTS).index

        try:
            self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index
        except Exception:
            self.metadata_index = {
                "document_count": len(self.documents_index),
                "average_document_length": {
                    field.value: (
                        sum(self.document_lengths_index[field].values()) / len(self.document_lengths_index[field])
                        if len(self.document_lengths_index[field]) > 0 else 0.0
                    )
                    for field in self.fields
                },
            }

        self.preprocessor = Preprocessor()


    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        Search for documents relevant to the query.

        Input:
            query (str | list): Input query as raw text or token list.
            method (str): Retrieval method.
            weights (dict): Weight of each field in final ranking.
            safe_ranking (bool): Whether to use full-index ranking.
            max_results (int | None): Maximum number of returned results.
            smoothing_method (str | None): Smoothing method for unigram model.
            alpha (float): Bayesian smoothing parameter.
            lamda (float): Mixture smoothing parameter.

        Output:
            list: Ranked list of (document_id, score) tuples.

        Function:
            Preprocesses the query, computes scores for each field using the
            selected retrieval approach, aggregates the field scores, and returns
            the ranked result list.
        """
        preprocessed_query = self.preprocessor.preprocess_text(query)

        final_scores = {}

        if safe_ranking:
            if method in ("ltn.lnn", "ltc.lnc", "OkapiBM25"):
                self.find_scores_with_safe_ranking(preprocessed_query, method, weights, final_scores)
            else:
                self.find_scores_with_unigram_model(preprocessed_query, smoothing_method, weights, final_scores, alpha, lamda)
        else:
            self.find_scores_with_unsafe_ranking(preprocessed_query, method, weights, max_results, final_scores, smoothing_method, alpha, lamda)

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        if max_results > 0:
            ranked = ranked[:max_results]

        return ranked


    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of different fields.
        """
        for field, weight in weights.items():
            if weight == 0 or field not in scores:
                continue
            for doc_id, score in scores[field].items():
                final_scores[doc_id] = final_scores.get(doc_id, 0.0) + weight * score


    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores,
        smoothing_method=None, alpha=0.5, lamda=0.5):
        """
        Compute scores using tiered indexes.

        Input:
            query (list): Tokenized query.
            method (str): Retrieval method.
            weights (dict): Field weights.
            max_results (int | None): Maximum number of results to consider.
            scores (dict): Output dictionary for scores.

        Output:
            None

        Function:
            Computes document scores using the tiered index structure.
        """
        tier_names = ["first_tier", "second_tier", "third_tier"]
        N = len(self.documents_index)

        field_seen = {field: {} for field in self.fields if weights.get(field, 0) > 0}
        global_scored_docs = set()      # documents already added to final scores

        for tier in tier_names:
            for field in list(field_seen.keys()):
                tier_index = self.tiered_index[field].get(tier, {})
                if not tier_index:      # empty tier
                    continue

                scorer = Scorer(tier_index, N)

                if method in ("ltn.lnn", "ltc.lnc"):
                    doc_scores = scorer.compute_scores_with_vector_space_model(query, method)
                elif method == "OkapiBM25":
                    avg_length = self._get_average_length(field)
                    doc_lengths = self.document_lengths_index[field]
                    doc_scores = scorer.compute_scores_with_okapi_bm25(
                        query, avg_length, doc_lengths
                    )
                else:
                    doc_lengths = self.document_lengths_index[field]
                    doc_scores = scorer.compute_scores_with_unigram_model(query, smoothing_method, doc_lengths, alpha, lamda)

                for doc_id, score in doc_scores:
                    if doc_id not in field_seen[field]:
                        field_seen[field][doc_id] = score
                        weight = weights[field]
                        scores[doc_id] = scores.get(doc_id, 0.0) + weight * score
                        global_scored_docs.add(doc_id)

            # we have enough distinct documents
            if max_results > 0 and len(global_scored_docs) >= max_results:
                break


    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Compute scores using the full indexes.

        Input:
            query (list): Tokenized query.
            method (str): Retrieval method.
            weights (dict): Field weights.
            scores (dict): Output dictionary for scores.

        Output:
            None

        Function:
            Computes document scores using the complete index of each field.
        """
        N = len(self.documents_index)
        field_scores = {}

        for field in self.fields:
            if weights.get(field, 0) == 0:
                continue

            index = self.document_indexes[field]
            scorer = Scorer(index, N)
            if method in ("ltn.lnn", "ltc.lnc"):
                doc_scores = scorer.compute_scores_with_vector_space_model(query, method)
            else:
                avg_length = self._get_average_length(field)
                doc_lengths = self.document_lengths_index[field]
                doc_scores = scorer.compute_scores_with_okapi_bm25(query, avg_length, doc_lengths)

            field_scores[field] = dict(doc_scores)

        self.aggregate_scores(weights, field_scores, scores)


    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Compute scores using the unigram language model.

        Input:
            query (list): Tokenized query.
            smoothing_method (str): Selected smoothing method.
            weights (dict): Field weights.
            scores (dict): Output dictionary for scores.
            alpha (float): Bayesian smoothing parameter.
            lamda (float): Mixture smoothing parameter.

        Output:
            None

        Function:
            Computes document scores for each field using a unigram language model.
        """
        N = len(self.documents_index)
        field_scores = {}

        for field in self.fields:
            if weights.get(field, 0) == 0:
                continue

            index = self.document_indexes[field]
            scorer = Scorer(index, N)
            doc_lengths = self.document_lengths_index[field]
            doc_scores = scorer.compute_scores_with_unigram_model(query, smoothing_method, doc_lengths, alpha, lamda)

            field_scores[field] = dict(doc_scores)

        self.aggregate_scores(weights, field_scores, scores)


    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.
        """
        merged = dict(scores1)
        for doc_id, score in scores2.items():
            merged[doc_id] = merged.get(doc_id, 0.0) + score
        return merged


    def _get_average_length(self, field):
        avg_lengths = self.metadata_index.get("average_document_length", {})
        if field.value in avg_lengths:
            return avg_lengths[field.value]
        if field in avg_lengths:
            return avg_lengths[field]
        lengths = self.document_lengths_index[field]
        return sum(lengths.values()) / len(lengths) if len(lengths) > 0 else 0.0


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "magic adventure"
    method = "lnc.ltc"
    weights = {
        Indexes.CHARACTERS: 1,
        Indexes.GENRES: 1,
        Indexes.DESCRIPTIONS: 1,
    }
    result = search_engine.search(query, method, weights)
    print(result)