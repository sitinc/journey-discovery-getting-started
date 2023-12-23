# MIT License
#
# Copyright (c) 2023 Smart Interactive Transformations Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict
import pandas as pd
import codecs
import os
import re
import spacy
import matplotlib.pyplot as plt

from tsdiscovery.openaiwrap import OpenAiWrap, CreateCompletions
from tsdiscovery.clusterwrap import ClusterWrap
from tsdiscovery.utils import Utils


sys_prompt_gen_transcript = """You are helping me generate example transcripts.  Do not reply back with anything other 
than the transcript content itself.  No headers or footers.  Separate each turn with a blank line.  Each line should 
start with either "USER: " or "AGENT: "."""


def count_lines(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as file:
        return sum(1 for line in file)


class Transcripts:
    """
    Utility class for working with transcript generation and processing.
    """

    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 spacy_nlp: spacy.language,
                 max_transcripts: int = 2000):
        """
        Construct a new instance.

        :param openai: The OpenAiWrap client instance.
        :param spacy_nlp: The spaCy language model instance.
        :param max_transcripts: The maximum number of transcripts that can be generated by gen_agent_transcripts.
        """
        self.openai = openai
        self.spacy_nlp = spacy_nlp
        self.max_transcripts = max_transcripts

    def gen_agent_transcript(self,
                             *,
                             user_prompt: str,
                             model: str,
                             session_id: str = None
                             ):
        """
        Generate an agent transcript.
        :param model: The OpenAI chat completion model.
        :param user_prompt: The chat completion user prompt.
        :param session_id: The session ID.
        :return: the result transcript.
        """
        if session_id is None:
            session_id = Utils.new_session_id()

        cmd = CreateCompletions(
            session_id=session_id,
            model=model,
            sys_prompt=sys_prompt_gen_transcript,
            user_prompt=user_prompt,
        )
        return self.openai.execute(cmd).result

    def gen_agent_transcripts(self,
                              *,
                              user_prompt: str,
                              session_id: str = None,
                              quantity: int = 5,
                              model: str = "gpt-4-1106-preview",
                              output_dir: str = "output"
                              ) -> None:
        """
        Generate a series of agent transcripts and output them to files.
        :param user_prompt: The chat completion user prompt.
        :param session_id: The session ID.
        :param quantity: The number of transcripts to generate.
        :param model: The OpenAI chat completion model.  Default is "gpt-4-1106-preview"
        :param output_dir: The transcript file output directory.
        :return: transcripts will be output to file system.
        """
        if quantity > self.max_transcripts:
            raise Exception(f"max quantity is {self.max_transcripts} unless you set max_transcripts via constructor")

        for i in range(0, quantity):
            final_file_name = f'{output_dir}/transcript{i}.txt'
            os.makedirs(output_dir, exist_ok=True)

            if os.path.exists(final_file_name):
                continue

            gen_transcript = self.gen_agent_transcript(
                session_id=session_id,
                model=model,
                user_prompt=user_prompt,
            )
            with codecs.open(final_file_name, 'w', 'utf-8') as f:
                f.write(gen_transcript)

    @staticmethod
    def concat_transcripts(dir_name: str, file_name: str) -> bool:
        """
        Concatenate transcripts
        :param dir_name: The directory with transcripts.
        :param file_name: The name of the combined transcripts file.
        :return: True if no errors occurred.
        """
        combined_name = f"{dir_name}/{file_name}"
        for file in os.listdir(dir_name):
            ts_file_name = os.path.join(dir_name, file)
            if os.path.isfile(ts_file_name):
                with codecs.open(ts_file_name, 'r', 'utf-8') as rf:
                    lines = rf.read()
                    lines = lines + '\n\n'

                with codecs.open(combined_name, 'a+', 'utf-8') as wf:
                    wf.write(lines)
        return True

    def split_sentences(self, utterance: str) -> list[str]:
        doc = self.spacy_nlp(utterance)

        utterance_sentences = [sent.text for sent in doc.sents]

        utterances_lines = []
        for sentence in utterance_sentences:
            if len(sentence.strip()) != 0:
                utterances_lines.append(sentence)

        return utterances_lines

    def process_transcript_to_csv(self, file_name: str) -> bool:
        try:
            csv_lines = ["participant,utterance"]
            invalid_lines = []
            csv_file = re.sub("\\.txt", ".csv", file_name)
            with codecs.open(file_name, 'r', 'utf-8') as f:
                lines = f.readlines()

            for line in lines:
                valid_line = re.search("^(USER|AGENT): (.*)\\.?$", line)
                if valid_line is None:
                    empty_line = re.search("\\r?\\n", line)
                    if empty_line is None:
                        invalid_lines.append(line)
                else:
                    participant = valid_line.group(1)
                    utterances = valid_line.group(2)

                    utterances = re.sub(",", "", utterances)

                    if re.search("\\.\\s*", utterances) is not None:
                        utterances_lines = self.split_sentences(utterances)

                        for final_line in utterances_lines:
                            csv_line = participant + ',' + final_line
                            csv_lines.append(csv_line)
                    else:
                        csv_line = participant + ',' + utterances
                        csv_lines.append(csv_line)

            csv_text = "\n".join(csv_lines)

            with codecs.open(csv_file, 'w', 'utf-8') as csv:
                csv.write(csv_text)

            if len(invalid_lines) > 0:
                print(invalid_lines)
                return False
            return True
        except UnicodeDecodeError as err:
            print(f"Error processing file: {file_name}: {err.reason}")
            return False

    @staticmethod
    def get_transcript_utterances(*,
                                  file_name: str,
                                  col_name: str,
                                  remove_dups: bool = True) -> list[str]:
        """Get utterances from a CSV file."""
        # Load the CSV file to a data frame.
        df = pd.read_csv(file_name)
        # Get the named column as a list.
        utterances = df[col_name].tolist()

        # Remove duplicates, default behaviour.
        if remove_dups:
            utterances_set = set(utterances)
            utterances = list(utterances_set)

        return utterances

    def extract_entities(self, file_path: str, output_dir: str):
        entities_by_type = defaultdict(set)

        with codecs.open(file_path, 'r', 'utf-8') as file:
            for line in file:
                doc = self.spacy_nlp(line)
                for ent in doc.ents:
                    # Add the entity text to the set corresponding to its type
                    entities_by_type[ent.label_].add(ent.text)

        # Create a directory to store the entity files
        os.makedirs(output_dir, exist_ok=True)

        # Create and write to files for each entity type
        for entity_type, examples in entities_by_type.items():
            file_name = f"{output_dir}/{entity_type.lower().replace(' ', '')}.txt"
            with codecs.open(file_name, 'w', 'utf-8') as f:
                for example in examples:
                    f.write(example + '\n')

    def cluster_and_name_utterances(self,
                                    *,
                                    csv_file: str,
                                    csv_col: str,
                                    output_dir: str,
                                    min_cluster_size=40,
                                    min_samples=5,
                                    epsilon=0.2,
                                    ) -> None:
        # Initialize the Cluster Client.
        cluster_client = ClusterWrap(
            openai=self.openai,
            spacy_nlp=self.spacy_nlp,
            embeddings_model='MiniLM',
        )

        # Get the utterances.
        utterances = Transcripts.get_transcript_utterances(
            file_name=csv_file,
            col_name=csv_col,
        )

        # Create the embeddings.
        embeddings = cluster_client.get_embeddings(utterances=utterances)

        # Reduce dimensionality.
        umap_embeddings = cluster_client.reduce_dimensionality(embeddings=embeddings)

        # Cluster the utterances.
        cluster = cluster_client.hdbscan(
            embeddings=umap_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon,
        )

        labels = cluster.labels_

        # Get silhouette score.
        silhouette_avg = cluster_client.get_silhouette(umap_embeddings, cluster)
        print(f'Silhouette Score: {silhouette_avg:.2f}')

        clustered_sentences = cluster_client.get_clustered_sentences(utterances, cluster)

        new_labels = cluster_client.get_new_cluster_labels(
            clustered_sentences=clustered_sentences,
            output_dir=output_dir,
        )

        cluster_client.visualize_clusters(umap_embeddings, labels, new_labels)

    @staticmethod
    def visualize_intent_bars(directory: str):
        file_names = []
        line_counts = []

        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                if file == "-1_noise.txt":
                    continue

                normed_file = re.sub("[0-9]+_(.*?)\\.txt", "\\1", file)
                file_names.append(normed_file)
                line_counts.append(count_lines(os.path.join(directory, file)))

        # Sort the files by line count in descending order
        sorted_data = sorted(zip(line_counts, file_names), reverse=True)
        line_counts_sorted, file_names_sorted = zip(*sorted_data)

        # Adjust the figure size to accommodate all entries
        plt.figure(figsize=(10, len(file_names_sorted) * 0.5))

        plt.barh(file_names_sorted, line_counts_sorted)
        plt.xlabel('Utterance Count')
        plt.title('Number of Unique Utterances per-Cluster (Sorted)')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
        plt.tight_layout()  # Adjust layout to fit all labels
        plt.show()

    @staticmethod
    def visualize_intent_pie(directory):
        file_names = []
        line_counts = []

        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                normed_file = re.sub("[0-9]+_(.*?)\\.txt", "\\1", file)
                file_names.append(normed_file)
                line_counts.append(count_lines(os.path.join(directory, file)))

        # Group slices below 1%
        total_lines = sum(line_counts)
        threshold = 0.01 * total_lines
        small_files = [count for count in line_counts if count < threshold]
        other_count = sum(small_files)

        # Filter out small files and add "Other" category
        line_counts_filtered = [count for count in line_counts if count >= threshold]
        file_names_filtered = [file_names[i] for i, count in enumerate(line_counts) if count >= threshold]

        if other_count > 0:
            line_counts_filtered.append(other_count)
            file_names_filtered.append("Other")

        # Sort the slices by size
        sorted_data = sorted(zip(line_counts_filtered, file_names_filtered), reverse=True)
        line_counts_sorted, file_names_sorted = zip(*sorted_data)

        # Adjust the figure size
        plt.figure(figsize=(12, 12))

        wedges, texts, autotexts = plt.pie(line_counts_sorted, autopct='%1.1f%%', startangle=140)

        # Equal aspect ratio ensures the pie chart is circular.
        plt.axis('equal')

        # Add a legend
        plt.legend(wedges, file_names_sorted, title="Files", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.title('Distribution of Unique Utterances by Cluster (<1% as Other, Sorted by Size)')
        plt.show()

    @staticmethod
    def visualize_entity_bars(directory: str):
        file_names = []
        line_counts = []

        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                scrub_file = re.sub("\\.txt", "", file)
                file_names.append(scrub_file)
                line_counts.append(count_lines(os.path.join(directory, file)))

        # Sort the files by line count in descending order
        sorted_data = sorted(zip(line_counts, file_names), reverse=True)
        line_counts_sorted, file_names_sorted = zip(*sorted_data)

        # Adjust the figure size to accommodate all entries
        plt.figure(figsize=(10, len(file_names_sorted) * 0.5))

        plt.barh(file_names_sorted, line_counts_sorted)
        plt.xlabel('Value Count')
        plt.title('Number of Values per-Entity (Sorted)')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
        plt.tight_layout()  # Adjust layout to fit all labels
        plt.show()

    @staticmethod
    def visualize_entity_pie(directory):
        file_names = []
        line_counts = []

        # Reading files and counting lines
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                scrub_file = re.sub("\\.txt", "", file)
                file_names.append(scrub_file)
                line_counts.append(count_lines(os.path.join(directory, file)))

        # Sorting the data by line counts in descending order
        sorted_data = sorted(zip(line_counts, file_names), reverse=True)
        line_counts_sorted, file_names_sorted = zip(*sorted_data)

        # Adjust the figure size
        plt.figure(figsize=(12, 12))

        # Creating the pie chart
        wedges, texts, autotexts = plt.pie(line_counts_sorted, autopct='%1.1f%%', startangle=140)

        # Equal aspect ratio ensures the pie chart is circular
        plt.axis('equal')

        # Adding a legend to the side of the chart
        plt.legend(wedges, file_names_sorted, title="Entity Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.title('Distribution of Values by Entity Type (Sorted by Size)')
        plt.show()

    @staticmethod
    def detail_entity_types(directory):
        file_names = []

        # Reading files and counting lines
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                scrub_file = re.sub("\\.txt", "", file)
                print(f"{scrub_file} - {spacy.explain(scrub_file.upper())}")
