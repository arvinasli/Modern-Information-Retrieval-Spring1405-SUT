import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes


    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        words = document.split()
        shingles = set()
        
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i+k])
            shingles.add(shingle)
        
        return shingles
    

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingle_sets = []
        all_shingles = set()

        for doc in self.documents:
            shingles = self.shingle_document(doc)
            doc_shingle_sets.append(shingles)
            all_shingles.update(shingles)
        
        sorted_shingles = sorted(all_shingles)
        shingle_to_idx = {shingle: i for i, shingle in enumerate(sorted_shingles)}
        
        matrix = np.zeros((len(sorted_shingles), len(self.documents)), dtype=int)
        for doc_id, shingles in enumerate(doc_shingle_sets):
            for shingle in shingles:
                matrix[shingle_to_idx[shingle], doc_id] = 1
        return matrix
    

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_matrix = self.build_characteristic_matrix()
        n_shingles, n_docs = characteristic_matrix.shape
        signature_matrix = np.full((self.num_hashes, n_docs), -1)

        for i in range(self.num_hashes):
            perm = np.random.permutation(n_shingles)
            for doc in range(n_docs):
                column_perm = characteristic_matrix[perm, doc]
                first_occurence = np.argmax(column_perm)
                signature_matrix[i, doc] = first_occurence
        
        return signature_matrix


    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        n_hashes, n_docs = signature.shape
        buckets = {}

        for band in range(bands):
            start = band * rows_per_band
            end = start + rows_per_band
            for doc_id in range(n_docs):
                band_slice = signature[start:end, doc_id]
                band_vector = tuple(band_slice.tolist())
                hashed = hash(band_vector)

                bucket_key = f"b{band}_{hashed}"

                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(doc_id)

        return buckets


    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes//num_bands)
        return ans


    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        if len(first_set | second_set) == 0:
            return 0.0
        return len(first_set & second_set) / len(first_set | second_set)


    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


def main():
    import json
    with open('Logic/LSHFakeData.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    documents = [" ".join(item["descriptions"]) for item in raw_data]

    num_hashes = 100
    lsh = MinHashLSH(documents, num_hashes)
    
    # run full LSH process (uses bands=25, rows_per_band=num_hashes//25=4)
    buckets = lsh.perform_lsh()
    
    lsh.jaccard_similarity_test(buckets, documents)

    # Map doc index -> original ID from raw_data
    doc_ids = [item["id"] for item in raw_data]

    print("\n--- Pairs found in the same bucket ---")
    for bucket_key, doc_indices in buckets.items():
        unique_docs = set(doc_indices)
        if len(unique_docs) > 1:
            for doc_i, doc_j in itertools.combinations(unique_docs, 2):
                # Compute Jaccard for this pair
                shingles_i = lsh.shingle_document(documents[doc_i], 2)
                shingles_j = lsh.shingle_document(documents[doc_j], 2)
                score = lsh.jaccard_score(shingles_i, shingles_j)
                print(f"{bucket_key} | Pair: ({doc_ids[doc_i]} , {doc_ids[doc_j]})  |  Jaccard = {score:.4f}")


if __name__ == '__main__':
    main()
