
search_url=https://serpapi.com/search
serp_api_key="" # put your serp api key here (https://serpapi.com/)

python search_r1/search/online_search_server.py --search_url $search_url \
                                            --topk 3 \
                                            --serp_api_key $serp_api_key
