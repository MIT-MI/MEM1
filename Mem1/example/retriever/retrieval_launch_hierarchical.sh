
file_path=/the/path/you/save/corpus
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
reranker_path=cross-encoder/ms-marco-MiniLM-L12-v2

python search_r1/search/retrieval_rerank_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --retrieval_topk 10 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --reranking_topk 3 \
                                            --reranker_model $reranker_path \
                                            --reranker_batch_size 32
