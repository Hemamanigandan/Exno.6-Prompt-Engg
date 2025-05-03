# Exno.6-Prompt-Engg
# Date:03-05-2025
# Register no:212222050055
# Aim: Development of Python Code Compatible with Multiple AI Tools



# Algorithm: 
1.  **Import Necessary Libraries:** Includes libraries for environment variables, JSON handling, abstract base classes, typing, and sentence similarity.
2.  **API Key Loading:** Retrieves API keys from environment variables for security. **Remember to set these in your environment.**
3.  **`AIAPIInterface` (Abstract Base Class):** Defines a common interface for all AI API interactions, ensuring that each API implementation has a `generate_response` method.
4.  **`OpenAIAPI` and `CohereAPI` (Concrete Implementations):** These classes implement the `AIAPIInterface` for specific AI APIs. They handle API-specific authentication and request/response formats.
    * They include error handling for missing libraries and API communication issues.
    * You'll need to install the respective API client libraries (`pip install openai cohere`).
    * **You can extend this by creating similar classes for other APIs like Anthropic.**
5.  **`compare_responses` Function:**
    * Takes a dictionary of API responses as input.
    * Uses the `sentence-transformers` library to generate embeddings for each response. Sentence embeddings capture the semantic meaning of the text.
    * Calculates the cosine similarity between the embeddings of different API responses. Cosine similarity is a measure of how similar two non-zero vectors are.
    * Returns a dictionary where keys are the API pairs being compared, and values are their similarity scores.
6.  **`analyze_comparison` Function:**
    * Takes the comparison scores and the original responses as input.
    * Analyzes the similarity scores to generate actionable insights:
        * **High Similarity:** Suggests consensus; recommends using either response or synthesizing them.
        * **Moderate Similarity:** Indicates some common ground but also differences; recommends reviewing both for nuances.
        * **Low Similarity:** Suggests divergent interpretations; recommends careful examination and potential prompt rephrasing.
    * Provides recommendations based on the level of similarity.
    * Includes snippets of the differing responses when similarity is low to aid in understanding the differences.
7.  **`main` Function:**
    * Takes a user prompt and a list of `AIAPIInterface` objects as input.
    * Iterates through the provided APIs, calls their `generate_response` method, and stores the responses in a dictionary.
    * Calls `compare_responses` to get the similarity scores.
    * Calls `analyze_comparison` to generate insights.
    * Organizes the prompt, responses, comparison scores, and insights into a structured dictionary.
    * Prints the results in a nicely formatted JSON.
8.  **`if __name__ == "__main__":` Block:**
    * This block executes when the script is run directly.
    * Sets a sample user prompt.
    * Initializes instances of the `OpenAIAPI` and `CohereAPI` (make sure you have your API keys set as environment variables).
    * Creates a list of the API clients to be used.
    * Calls the `main` function to start the automation process.

**To Use This Code:**

1.  **Install Required Libraries:**
    ```bash
    pip install openai cohere sentence-transformers scikit-learn
    ```
2.  **Set API Keys as Environment Variables:**
    * For OpenAI: `export OPENAI_API_KEY="your_openai_api_key"`
    * For Cohere: `export COHERE_API_KEY="your_cohere_api_key"`
    * Add others as needed.
3.  **Run the Python Script:**
    ```bash
    python your_script_name.py
    ```
# prompt:
You are an expert Python developer and AI systems integrator.
I need to automate the process of interacting with multiple AI APIs, comparing their outputs, and generating actionable insights based on those comparisons.
Please generate Python code that:

Accepts a user prompt and sends it to at least two different AI APIs (e.g., OpenAI, Cohere, Anthropic, etc.).

Collects and compares the outputs using a suitable similarity or comparison method.

Analyzes the comparison and generates actionable insights or recommendations based on the similarities or differences in the responses.

Organizes the results and insights in a clear, structured format (such as a dictionary or JSON).
The code should be modular, reusable, and easy to extend for more APIs or advanced analysis in the future.
Include comments to explain each step."
# code: 
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

class AIAPIInterface(ABC):
    """Abstract base class for interacting with different AI APIs."""
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class OpenAIAPI(AIAPIInterface):
    """Implementation for interacting with the OpenAI API."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        try:
            import openai
            openai.api_key = self.api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message['content']
        except ImportError:
            print("Error: OpenAI library not installed. Please install it using 'pip install openai'.")
            return None
        except Exception as e:
            print(f"Error communicating with OpenAI API: {e}")
            return None

class CohereAPI(AIAPIInterface):
    """Implementation for interacting with the Cohere API."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Cohere API key not found in environment variables.")
        try:
            import cohere
            co = cohere.Client(self.api_key)
            response = co.generate(
                model="command-r-plus",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            return response.generations[0].text
        except ImportError:
            print("Error: Cohere library not installed. Please install it using 'pip install cohere'.")
            return None
        except Exception as e:
            print(f"Error communicating with Cohere API: {e}")
            return None

def compare_responses(responses: Dict[str, str]) -> Dict[str, float]:
    """Compares the text responses using cosine similarity of their embeddings."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = list(responses.values())
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    api_names = list(responses.keys())
    comparison_scores = {}
    for i in range(len(api_names)):
        for j in range(i + 1, len(api_names)):
            score = similarity_matrix[i][j]
            comparison_scores[f"{api_names[i]} vs {api_names[j]}"] = float(score)
    return comparison_scores

def analyze_comparison(comparison_scores: Dict[str, float], responses: Dict[str, str]):
    """Analyzes the comparison scores and generates actionable insights."""
    insights = {}
    for comparison, score in comparison_scores.items():
        api1, api2 = comparison.split(" vs ")
        response1 = responses[api1]
        response2 = responses[api2]
        insights[comparison] = {
            "similarity_score": score
        }
        if score > 0.8:
            insights[comparison]["insight"] = "Responses are highly similar, suggesting consensus."
            insights[comparison]["recommendation"] = f"Consider either response as a reliable answer."
        elif 0.5 <= score <= 0.8:
            insights[comparison]["insight"] = "Responses show moderate similarity, indicating some differences."
            insights[comparison]["recommendation"] = f"Review both responses to identify the best fit."
        else:
            insights[comparison]["insight"] = "Responses are significantly different, suggesting divergent perspectives."
            insights[comparison]["recommendation"] = f"Carefully examine both responses before making a decision."
            insights[comparison][f"{api1}_response"] = response1[:200] + "..." if len(response1) > 200 else response1
            insights[comparison][f"{api2}_response"] = response2[:200] + "..." if len(response2) > 200 else response2
    return insights

def main(prompt: str, apis: List[AIAPIInterface]) -> Dict:
    """Main function to orchestrate the process."""
    responses = {}
    for api in apis:
        api_name = type(api).__name__.replace("API", "").lower()
        print(f"Calling {api_name}...")
        response = api.generate_response(prompt)
        if response is not None:
            responses[api_name] = response
            print(f"{api_name} response received.")
        else:
            print(f"Failed to get response from {api_name}.")
    if not responses:
        return {"error": "No responses received from any of the APIs."}
    print("\nComparing responses...")
    comparison_scores = compare_responses(responses)
    print("\nAnalyzing comparison...")
    insights = analyze_comparison(comparison_scores, responses)
    results = {
        "prompt": prompt,
        "responses": responses,
        "comparison_scores": comparison_scores,
        "insights": insights
    }
    print("\nResults:")
    print(json.dumps(results, indent=4))
    return results

if __name__ == "__main__":
    user_prompt = "What are the key benefits of using large language models in customer support automation?"
    openai_api = OpenAIAPI(api_key=OPENAI_API_KEY)
    cohere_api = CohereAPI(api_key=COHERE_API_KEY)
    apis_to_use = [openai_api, cohere_api]
    if any(api.api_key is None for api in apis_to_use):
        print("Warning: Some API keys are missing. Ensure environment variables are set.")
    else:
        main(user_prompt, apis_to_use)
# AI models used for verification :
  1. chat gpt
# output:
   ![image](https://github.com/user-attachments/assets/07cafaa0-4d43-4e12-895b-9c1341170ec8)
  2. gemini ai
# output:
![image](https://github.com/user-attachments/assets/aa00e876-13aa-465b-abd6-34b4db35e9be)

  3. perplexity ai
# output:

![image](https://github.com/user-attachments/assets/0884a553-7193-4bb6-84a8-7f21c9628df1)


# Result: The corresponding Prompt is executed in the three AI models successfully.
