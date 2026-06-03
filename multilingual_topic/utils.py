import os
import json
import csv
import pandas as pd
from tqdm.notebook import tqdm
import xml.etree.ElementTree as ET


class FileHandler():
    def __init__(self, project_folder: str):
        self.project_folder = project_folder

    # create file directory
    def create_filename(self, file_name):
        return f'{self.project_folder}/{file_name}'

    # create file directory alias
    def cr_fn(self, file_name):
        return self.create_filename(file_name)

    def csv_to_df(self, csvFilePath):
        jsonArray = []

        # read csv file
        with open(self.cr_fn(csvFilePath), encoding='utf-8') as csvf:
            # load csv file data using csv library's dictionary reader
            csvReader = csv.DictReader(csvf)

            # convert each csv row into python dict
            for row in tqdm(csvReader):
                # add this python dict to json array
                jsonArray.append(row)
        return pd.DataFrame(jsonArray)

    def df_to_csv(self, data, csvFilePath):
        data.to_csv(
            self.cr_fn(csvFilePath),
            index=False,
        )

    def df_to_json(self, data, filePath):
        data.to_json(
            self.cr_fn(filePath),
            orient='records',
            lines=True,
        )

    def json_to_df(self, filePath):
        data = pd.read_json(
            self.cr_fn(filePath),
            lines=True,
        )
        return data

    def read_tmx(self, file_path, lang):
        # Parse the TMX file
        tree = ET.parse(self.cr_fn(file_path))
        root = tree.getroot()
        # Initialize lists to store Spanish and English sentences
        sentences = []
        # Iterate through each translation unit
        for tu in root.findall('.//tu'):
            text = None
            for tuv in tu.findall('.//tuv'):
                language = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')
                if language == lang:
                    text = tuv.find('.//seg').text
                sentences.append(text)
        return sentences

    def read_text(self, file_path):
        sentences = []

        # Read files and store sentences
        with open(self.cr_fn(file_path), 'r', encoding='utf-8') as file:
            for line in file:
                sentences.append(line.strip())
        return sentences

    def read_parallel_corpus(self, file_path):
        with open(self.cr_fn(file_path), 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Initialize lists to store source, target1, and target2 texts
        source_texts = []
        target1_texts = []

        # Iterate over the lines and extract texts
        for i in range(0, len(lines), 4):  # Step over every 4 lines (source, target1, target2, blank)
            if i + 2 < len(lines):  # Ensure there are enough lines left for a full set
                source_texts.append(lines[i].strip())
                target1_texts.append(lines[i + 1].strip())

        # Combine the texts into a list of tuples
        parallel_corpus = list(zip(source_texts, target1_texts))
        return parallel_corpus

    def read_json(self, file_path):
        with open(self.cr_fn(file_path), 'r') as file:
            stored_annotations = json.load(file)
        return stored_annotations

    def save_json(self, data, file_path):
        with open(self.cr_fn(file_path), 'w') as file:
            json.dump(data, file, indent=4)  # Pretty print with 4 spaces indentation

    def create_folder(self, folder_path):
        # Create the folder
        if not os.path.exists(self.cr_fn(folder_path)):
            os.makedirs(self.cr_fn(folder_path))
            print(f"Folder '{folder_path}' created successfully.")
        else:
            print(f"Folder '{folder_path}' already exists.")



