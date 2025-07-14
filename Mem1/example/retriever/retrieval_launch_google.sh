
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python search_r1/search/internal_google_server.py --api_key $api_key \
                                            --topk 5 \
                                            --cse_id $cse_id \
                                            --snippet_only
