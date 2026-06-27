from typing import Dict, List
from Logic.Search import SearchEngine
from Logic.spell_correction import SpellCorrection
from Logic.indexer.indexes_enum import Indexes
import json

data_path = 'crawled.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

books_dataset = {}
for book in data:
    books_dataset[book['id']] = book

# build corpus for spell correction once
corpus = [
    (book.get('description') if isinstance(book.get('description'), str) else "") + " " +
    ' '.join(book.get('genres', [])) + " " +
    ' '.join(book.get('characters', []))
    for book in data
]

# initialize tools
import os
index_dir = 'indexes'
if not os.path.exists(index_dir):
    os.makedirs(index_dir)
spell_pkl_path = os.path.join(index_dir, 'spell_correction.pkl')

spell_correction_obj = SpellCorrection(
    all_documents=corpus,
    load_path=spell_pkl_path,
    save_path=spell_pkl_path
)

search_engine = SearchEngine()

def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    safe_ranking: bool = True,
    smoothing_method = None,
    alpha: float = 0.5,
    lamda: float = 0.5,
    should_print=False,
    preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = {
        Indexes.CHARACTERS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.DESCRIPTIONS: weights[2],
    }
    return search_engine.search(
        query, method, weights, safe_ranking=safe_ranking, max_results=max_result_count, smoothing_method=smoothing_method, alpha=alpha, lamda=lamda
    )

def get_book_by_id(id: str, books_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get book by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the book

    books_dataset: List[Dict[str, str]]
        The dataset of books

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The book with the given id
    """
    result = books_dataset.get(id)

    return result
