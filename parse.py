# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

import os

def list_files_in_directory(directory):
  filenames = os.listdir(directory)
  for filename in filenames:
      print(filename)

directory_path = 'data/pdf/CS'
list_files_in_directory(directory_path)

print(os.getenv("LLAMA_CLOUD_API_KEY"))