from .indexes_enum import Indexes, Index_types
import json
import os


class Index_reader:
    def __init__(self, path: str, index_name: Indexes, index_type: Index_types = None):
        """
        Initializes the Index_reader.

        Parameters
        ----------
        path : str
            The path to the indexes.
        index_name : Indexes
            The name of the index to read.
        index_type : Index_types
            The type of the index to read.  
        """
        self.path = path
        self.index_name = index_name
        self.index_type = index_type
        self.index = self.get_index()


    def get_index(self):
        """
        Gets the index from the file.

        Returns
        -------
        dict
            The index.
        """
        base_name = self.index_name.value
        if self.index_type is not None:
            base_name += '_' + self.index_type.value
        file_name = base_name + '_index.json'
        absolute_path = os.path.join(self.path, file_name)
        
        with open(absolute_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        