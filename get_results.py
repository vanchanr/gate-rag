import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('LLAMA_CLOUD_API_KEY')}"}
base_url = "https://api.cloud.llamaindex.ai/api/parsing"
result_type = "markdown"

def download_results(subject):
    with open(f"data/job_ids/{subject}.json") as f:
        job_ids = json.load(f)

    for filename, job_id in job_ids.items():
        outfilename = filename.replace(".pdf", ".md")
        result_url = f"{base_url}/job/{job_id}/result/{result_type}"
        response = requests.get(result_url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            output = result[result_type]

            with open(f"data/md/{subject}/{outfilename}", "w") as f:
                f.write(output)
        else:
            print(f"Error for {filename}, {job_id}: {response.text}")

if __name__ == "__main__":
    subject = sys.argv[1]
    os.makedirs(f"data/md/{subject}", exist_ok=True)
    download_results(subject)
