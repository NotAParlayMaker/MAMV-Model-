from pathlib import Path
from mamv_model import MAMVModel
from mamv_model.document_qa import Answer
from mamv_model.ingestion import ingest_documents, chunk_documents
from mamv_model.retrieval import RetrievedDocument, RetrievalDiversitySettings, select_diverse

class Backend:
    def answer(self, document, question, **kwargs):
        return Answer('The capacity is 10.' if '10' in document else 'The capacity is 12.')

def test_collection_ids_provenance_duplicates_and_frame(tmp_path):
    a, b, copy = (tmp_path / n for n in ('a.txt', 'b.txt', 'copy.txt'))
    a.write_text('The capacity is 10.'); b.write_text('The capacity is 12.'); copy.write_text(a.read_text())
    collection, texts = ingest_documents([a, b, copy])
    assert len(collection.documents) == 2 and collection.collection_id.startswith('collection-')
    assert {c.document_id for c in chunk_documents(collection, texts)} == {d.document_id for d in collection.documents}
    answer = MAMVModel(Backend()).answer_files([a, b], 'capacity?', synthesis_mode='contradiction_first')
    assert answer.inference_frame.collection_id == collection.collection_id
    assert answer.synthesis_mode == 'contradiction_first' and answer.contradiction_candidates

def test_diverse_retrieval_caps_and_deduplicates():
    items = [RetrievedDocument('same', 'a1', .9, 'a', content_hash='x'), RetrievedDocument('same', 'a2', .8, 'a', content_hash='x'), RetrievedDocument('b', 'b1', .7, 'b')]
    selected, dropped, decisions = select_diverse(items, RetrievalDiversitySettings(top_k=3, max_chunks_per_document=1, min_documents=2))
    assert {x.document_id for x in selected} == {'a', 'b'} and len(dropped) == 1 and decisions
