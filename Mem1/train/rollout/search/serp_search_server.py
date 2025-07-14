import json
import os
import requests
import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import argparse
import uvicorn

parser = argparse.ArgumentParser(description="Launch online search server.")
parser.add_argument('--search_url', type=str, required=True, 
                    help="URL for search engine (e.g. https://serpapi.com/search)")
parser.add_argument('--topk', type=int, default=3, 
                    help="Number of results to return per query")
parser.add_argument('--serp_api_key', type=str, default=None, 
                    help="SerpAPI key for online search")
parser.add_argument('--serp_engine', type=str, default="google", 
                    help="SerpAPI engine for online search")
parser.add_argument('--requests_per_second', type=int, default=50,
                    help="Maximum number of requests per second")
parser.add_argument('--max_retries', type=int, default=3,
                    help="Maximum number of retry attempts for failed requests")
args = parser.parse_args()

# --- Config ---
class OnlineSearchConfig:
    def __init__(
        self,
        search_url: str = "https://serpapi.com/search",
        topk: int = 3,
        serp_api_key: Optional[str] = None,
        serp_engine: Optional[str] = None,
        requests_per_second: int = 50,
        max_retries: int = 3,
    ):
        self.search_url = search_url
        self.topk = topk
        self.serp_api_key = serp_api_key
        self.serp_engine = serp_engine
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries


# --- Online Search Wrapper ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str):
        payload = json.dumps({
            "q": query
        })
        headers = {
            "X-API-KEY": self.config.serp_api_key,
            "Content-Type": "application/json",
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(self.config.search_url, headers=headers, data=payload)
                response.raise_for_status()  # Raise exception for 4XX/5XX status codes
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 1.0
                    print(f"Request failed: {str(e)}. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{self.config.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"All retry attempts failed for query: {query}")
                    raise
        
        # This should not be reached due to the raise in the except block
        raise Exception("Max retry attempts exceeded")

    def batch_search(self, queries: List[str]):
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(self._search_query, query): query for query in queries}
            for future in future_to_query:
                try:
                    result = future.result()
                    results.append(self._process_result(result))
                except Exception as exc:
                    print(f"Query {future_to_query[future]} generated an exception: {exc}")
                    results.append([])
        return results

    def _process_result(self, search_result: Dict):
        results = []
        
        answer_box = search_result.get('answer_box', {})
        if answer_box:
            title = answer_box.get('title', 'No title.')
            snippet = answer_box.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        organic_results = search_result.get('organic', [])
        for _, result in enumerate(organic_results[:self.config.topk]):
            title = result.get('title', 'No title.')
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        related_results = search_result.get('related_questions', [])
        for _, result in enumerate(related_results[:self.config.topk]):
            title = result.get('question', 'No title.')  # question is the title here
            snippet = result.get('snippet', 'No snippet available.')
            results.append({
                'document': {"contents": f'\"{title}\"\n{snippet}'},
            })

        return results


# --- FastAPI Setup ---
app = FastAPI(title="Online Search Proxy Server")

class SearchRequest(BaseModel):
    queries: List[str]

# Instantiate global config + engine
config = OnlineSearchConfig(
    search_url=args.search_url,
    topk=args.topk,
    serp_api_key=args.serp_api_key,
    serp_engine=args.serp_engine,
    requests_per_second=args.requests_per_second,
    max_retries=args.max_retries,
)
engine = OnlineSearchEngine(config)

# --- Routes ---
@app.post("/retrieve")
def search_endpoint(request: SearchRequest):
    results = engine.batch_search(request.queries)
    return {"result": results}

## return {"result": List[List[{'document': {"id": xx, "content": "title" + \n + "content"}, 'score': xx}]]}

if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8012)
