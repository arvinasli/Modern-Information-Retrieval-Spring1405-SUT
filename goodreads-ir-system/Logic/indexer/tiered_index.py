from .indexes_enum import Indexes, Index_types
from .index_reader import Index_reader
import json
import os


class Tiered_index:
    def __init__(self, path="index/"):
        """
        Initializes the Tiered_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """

        self.index = {
            Indexes.CHARACTERS: Index_reader(path, index_name=Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(path, index_name=Indexes.DESCRIPTIONS).index,
        }

        # for ratings
        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index

        # precompute a static quality score g(d) for each document
        self.static_score = {}
        for doc_id, doc in self.documents_index.items():
            rating = doc.get("avg_rating", 0)
            if isinstance(rating, str):
                try:
                    rating = float(rating)
                except ValueError:
                    rating = 0.0
            self.static_score[doc_id] = rating

        # feel free to change the thresholds
        self.tiered_index = {
            Indexes.CHARACTERS: self.convert_to_tiered_index(3, 2, Indexes.CHARACTERS),
            Indexes.DESCRIPTIONS: self.convert_to_tiered_index(10, 5, Indexes.DESCRIPTIONS),
            Indexes.GENRES: self.convert_to_tiered_index(1, 0, Indexes.GENRES)
        }
        self.store_tiered_index(path, Indexes.CHARACTERS)
        self.store_tiered_index(path, Indexes.DESCRIPTIONS)
        self.store_tiered_index(path, Indexes.GENRES)

    def convert_to_tiered_index(
        self, first_tier_threshold: int, second_tier_threshold: int, index_name
    ):
        """
        Convert the current index to a tiered index.

        Parameters
        ----------
        first_tier_threshold : int
            The threshold for the first tier
        second_tier_threshold : int
            The threshold for the second tier
        index_name : Indexes
            The name of the index to read.

        Returns
        -------
        dict
            The tiered index with structure of 
            {
                "first_tier": dict,
                "second_tier": dict,
                "third_tier": dict
            }
        """
        if index_name not in self.index:
            raise ValueError("Invalid index type!")

        current_index = self.index[index_name]
        
        first_tier = {}
        second_tier = {}
        third_tier = {}

        for term, postings in current_index.items():
            # build list of (doc_id, combined_score) for this term
            scored_docs = []
            for doc_id, tf in postings.items():
                score = tf * self.static_score.get(doc_id, 0.0)
                scored_docs.append((doc_id, score))

            # sort by combined score
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            tier1 = scored_docs[:first_tier_threshold]
            tier2 = scored_docs[first_tier_threshold:first_tier_threshold + second_tier_threshold]
            tier3 = scored_docs[first_tier_threshold + second_tier_threshold:]

            if tier1:
                first_tier[term] = {doc_id: postings[doc_id] for doc_id, _ in tier1}
            if tier2:
                second_tier[term] = {doc_id: postings[doc_id] for doc_id, _ in tier2}
            if tier3:
                third_tier[term] = {doc_id: postings[doc_id] for doc_id, _ in tier3}

        return {
            "first_tier": first_tier,
            "second_tier": second_tier,
            "third_tier": third_tier,
        }

    def store_tiered_index(self, path, index_name):
        """
        Stores the tiered index to a file.
        """
        file_name = index_name.value + "_" + Index_types.TIERED.value + "_index.json"
        full_path = os.path.join(path, file_name)

        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(self.tiered_index[index_name], file, indent=2)


if __name__ == "__main__":
    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    indexes_path = os.path.join(project_root, 'indexes')
    tiered_index = Tiered_index(indexes_path)

    print('Tiered index stored successfully.')
