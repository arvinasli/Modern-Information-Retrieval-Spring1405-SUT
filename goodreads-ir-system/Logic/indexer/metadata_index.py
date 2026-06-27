from .index_reader import Index_reader
from .indexes_enum import Indexes, Index_types
import json
import os


class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.path = path
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)


    def read_documents(self, path):
        """
        Reads the documents.
        
        """
        return Index_reader(path, index_name=Indexes.DOCUMENTS).index


    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'characters': self.get_average_document_field_length('characters'),
            'genres': self.get_average_document_field_length('genres'),
            'descriptions': self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    
    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        if where == Indexes.CHARACTERS.value:
            field_index = Indexes.CHARACTERS
        elif where == Indexes.GENRES.value:
            field_index = Indexes.GENRES
        elif where == Indexes.DESCRIPTIONS.value:
            field_index = Indexes.DESCRIPTIONS
        else:
            raise ValueError(f"Unknown field: {where}")
        
        reader = Index_reader(self.path, index_name=field_index, index_type=Index_types.DOCUMENT_LENGTH)
        length_dic = reader.index

        total_len = 0
        n_docs = len(self.documents)

        for doc_id in self.documents.keys():
            total_len += length_dic.get(doc_id, 0)

        return total_len / n_docs


    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        file_name = Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        full_path = os.path.join(path, file_name)

        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(self.metadata_index, file, indent=2)

    
if __name__ == "__main__":
    # meta_index = Metadata_index('../../indexes/')

    # Get the directory where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    indexes_path = os.path.join(project_root, 'indexes')
    metadata_index = Metadata_index(indexes_path)

    print('Metadata index stored successfully.')