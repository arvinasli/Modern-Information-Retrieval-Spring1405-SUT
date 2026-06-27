import json
import os
from .indexes_enum import Indexes, Index_types
from .index_reader import Index_reader

class DocumentLengthsIndex:
    def __init__(self, path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """

        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.characters_index = Index_reader(path, Indexes.CHARACTERS).index
        self.genres_index = Index_reader(path, Indexes.GENRES).index
        self.descriptions_index = Index_reader(path, Indexes.DESCRIPTIONS).index

        self.document_length_index = {
            Indexes.CHARACTERS: self.get_documents_length(Indexes.CHARACTERS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS: self.get_documents_length(Indexes.DESCRIPTIONS.value)
        }
        self.store_document_lengths_index(path, Indexes.CHARACTERS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.DESCRIPTIONS)


    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        if where == Indexes.CHARACTERS.value:
            field_index = self.characters_index
        elif where == Indexes.GENRES.value:
            field_index = self.genres_index
        elif where == Indexes.DESCRIPTIONS.value:
            field_index = self.descriptions_index
        else:
            raise ValueError(f"Unknown field: {where}")
        
        doc_lengths = {}
        for term, postings in field_index.items():
            for doc_id, tf in postings.items():
                doc_lengths[doc_id] = doc_lengths.get(doc_id, 0) + tf

        return doc_lengths

    
    def store_document_lengths_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        file_name = index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        full_path = os.path.join(path, file_name)

        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(self.document_length_index[index_name], file, indent=2)
    

if __name__ == '__main__':
    # document_lengths_index = DocumentLengthsIndex('../../indexes/')

    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    indexes_path = os.path.join(project_root, 'indexes')
    document_lengths_index = DocumentLengthsIndex(indexes_path)

    print('Document lengths index stored successfully.')