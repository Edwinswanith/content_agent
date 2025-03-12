from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
import os

class PerplexitySearchInput(BaseModel):
    """Input schema for Perplexity Search."""
    query: str = Field(..., description="The search query to look up information about.")

class PerplexitySearchTool(BaseTool):
    name: str = "Perplexity Search"
    description: str = (
        "A powerful search tool that uses Perplexity API to find detailed "
        "and academic information about any topic. Particularly useful for "
        "finding research papers, technical information, and recent developments."
    )
    args_schema: Type[BaseModel] = PerplexitySearchInput

    def _run(self, query: str) -> str:
        """Execute the Perplexity search."""
        print("Searching with Perplexity...")
        
        url = "https://api.perplexity.ai/search"
        headers = {
            "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "max_results": 5,
            "include_citations": True
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            results = response.json()
            if not results.get('results'):
                return "Sorry, I couldn't find any results for that query."
            
            formatted_results = []
            for result in results['results']:
                formatted_result = '\n'.join([
                    f"Title: {result.get('title', 'No title')}",
                    f"Link: {result.get('url', 'No link')}",
                    f"Summary: {result.get('summary', 'No summary')}",
                    f"Citations: {', '.join(result.get('citations', ['No citations']))}"
                    "\n-----------------"
                ])
                formatted_results.append(formatted_result)
            
            return '\n'.join(formatted_results)
            
        except requests.exceptions.RequestException as e:
            return f"An error occurred while searching: {str(e)}" 