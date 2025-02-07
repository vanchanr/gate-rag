import os
import sys
import mimetypes
import requests
import json
from dotenv import load_dotenv

load_dotenv()
headers = {"Authorization": f"Bearer {os.getenv('LLAMA_CLOUD_API_KEY')}"}
base_url = "https://api.cloud.llamaindex.ai/api/parsing"

def start_parsing(subject):
    input_directory_path = f"data/pdf/{subject}"
    job_ids = {}

    for filename in os.listdir(input_directory_path):
        file_path = os.path.join(input_directory_path, filename)
        with open(file_path, "rb") as f:
            mime_type = mimetypes.guess_type(file_path)[0]
            files = {"file": (f.name, f, mime_type)}

            # send the request, upload the file
            url = f"{base_url}/upload"
            response = requests.post(url, headers=headers, files=files)

            response.raise_for_status()
            # get the job id for the result_url
            job_id = response.json()["id"]
            job_ids[filename] = job_id

    print(job_ids)
    with open(f"data/job_ids/{subject}.json", "w") as f:
        json.dump(job_ids, f)

if  __name__ == "__main__":
    subject = sys.argv[1]
    os.makedirs("data/job_ids", exist_ok=True)
    start_parsing(subject)
