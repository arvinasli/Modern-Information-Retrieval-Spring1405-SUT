from enum import Enum


class Indexes(Enum):
    DOCUMENTS = 'documents'
    CHARACTERS = 'characters'
    GENRES = 'genres'
    DESCRIPTIONS = 'description'

class Index_types(Enum):
    TIERED = 'tiered'
    DOCUMENT_LENGTH = 'document_length'
    METADATA = 'metadata'