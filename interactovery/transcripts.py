# MIT License
#
# Copyright (c) 2023 Justin Randall, Smart Interactive Transformations Inc.
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
import logging
import matplotlib.pyplot as plt
from collections import Counter

from interactovery.openaiwrap import OpenAiWrap, CreateCompletions
from interactovery.clusterwrap import ClusterWrap
from interactovery.utils import Utils

log = logging.getLogger('transcriptLogger')

sys_prompt_gen_transcript = """You are helping me generate example transcripts.  Do not reply back with anything other 
than the transcript content itself.  No headers or footers.  Only generate a single transcript example for each 
response.  Separate each turn with a blank line.  Each line should start with either "USER: " or "AGENT: "."""


def count_lines(file_path):
    with codecs.open(file_path, 'r', 'utf-8') as file:
        return sum(1 for line in file)


class Utterance:
    """
    Utterance metadata and content.
    """

    def __init__(self,
                 *,
                 source: str,
                 utterance: str,
                 count: int):
        """
        Construct a new instance.

        :param source: The source (file name, URL, etc.) of the utterance.
        :param utterance: The utterance.
        :param count: The utterance count.
        """
        self.source = source
        self.utterance = utterance
        self.count = count

    def __str__(self):
        return (f"Utterance(source={self.source}" +
                f", utterance={self.utterance}" +
                f", count={self.count}" +
                ")")

    def __repr__(self):
        return (f"Utterances(source={self.source!r}" +
                f", utterance={self.utterance!r}" +
                f", count={self.count!r}" +
                ")")


class Utterances:
    """
    Utterances metadata and content.
    """

    def __init__(self,
                 *,
                 source: str,
                 utterances: list[str]):
        """
        Construct a new instance.

        :param source: The source (file name, URL, etc.) of the utterances.
        :param utterances: The utterances.
        """
        self.source = source
        self.utterances = utterances
        utterance_count = len(utterances)
        self.utterance_count = utterance_count

        utterances_set = set(utterances)
        unique_utterances = list(utterances_set)
        self.unique_utterances = unique_utterances
        unique_utterance_count = len(unique_utterances)
        self.unique_utterance_count = unique_utterance_count

        unique_utterance_count_map = Counter(utterances)
        self.unique_utterance_count_map = unique_utterance_count_map

        unique_utterance_counts_values = sum(unique_utterance_count_map.values())

        if unique_utterance_counts_values != utterance_count:
            raise Exception(f'Consistency failure: ({unique_utterance_counts_values} != {utterance_count})')

    def __str__(self):
        return (f"Utterances(source={self.source}" +
                f", utterances=..." +
                f", utterance_count={self.utterance_count}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count}" +
                f", unique_utterance_count_map=..." +
                ")")

    def __repr__(self):
        return (f"Utterances(source={self.source!r}" +
                f", utterances=..." +
                f", utterances_count={self.utterance_count!r}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count!r}" +
                f", unique_utterance_count_map=..." +
                ")")


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

        if session_id is None:
            session_id = Utils.new_session_id()

        for i in range(0, quantity):
            final_file_name = f'{output_dir}/transcript{i}.txt'
            os.makedirs(output_dir, exist_ok=True)

            if os.path.exists(final_file_name):
                continue

            log.info(f"{session_id} | gen_agent_transcripts | Generating example transcript #{i}")
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

    def process_ts_lines_to_csv(self,
                                *,
                                source: str = None,
                                lines: list[str] = None) -> list[str]:
        csv_lines = []
        invalid_lines = []

        for ts_line in lines:
            valid_line = re.search("^(USER|AGENT): (.*)\\.?$", ts_line)
            if valid_line is None:
                empty_line = re.search("\\r?\\n", ts_line)
                if empty_line is None:
                    invalid_lines.append(ts_line)
            else:
                participant = valid_line.group(1)
                utterances = valid_line.group(2)

                utterances = re.sub(",", "", utterances)

                if re.search("\\.\\s*", utterances) is not None:
                    utterances_lines = self.split_sentences(utterances)

                    for final_line in utterances_lines:
                        csv_line = source + ',' + participant + ',' + final_line
                        csv_lines.append(csv_line)
                else:
                    csv_line = source + ',' + participant + ',' + utterances
                    csv_lines.append(csv_line)

        return csv_lines

    def concat_and_process_ts_to_csv(self,
                                     *,
                                     in_dir: str,
                                     out_dir: str,
                                     out_file: str) -> bool:
        """
        Concatenate and process the transcripts
        :param in_dir: The input directory with transcripts.
        :param out_dir: The output directory for the combined CSV.
        :param out_file: The name of the combined transcripts CSV output file.
        :return: True if no errors occurred.
        """
        try:
            combined_name = f"{out_dir}/{out_file}"

            os.makedirs(out_dir, exist_ok=True)

            with codecs.open(combined_name, 'w+', 'utf-8') as wf:
                wf.write("source,participant,utterance\n")

            ts_files = os.listdir(in_dir)

            file_progress = 0
            file_progress_total = len(ts_files)

            for ts_file in ts_files:
                file_progress = file_progress + 1
                Utils.progress_bar(file_progress, file_progress_total)
                try:
                    ts_file_path = os.path.join(in_dir, ts_file)
                    if os.path.isfile(ts_file_path):
                        with codecs.open(ts_file_path, 'r', 'utf-8') as rf:
                            lines = rf.readlines()

                            csv_lines = self.process_ts_lines_to_csv(
                                source=ts_file,
                                lines=lines,
                            )

                            csv_text = "\n".join(csv_lines)
                            csv_text = csv_text + '\n'

                            with codecs.open(combined_name, 'a+', 'utf-8') as wf:
                                wf.write(csv_text)
                except UnicodeDecodeError as err:
                    log.error(f"Unicode decode error for file: {ts_file}: {err.reason}")
                    return False
        except Exception as err:
            log.error(f"Unhandled exception: {err}")
            return False

        return True

    @staticmethod
    def get_transcript_utterances_for_party(*,
                                            party: str,
                                            file_name: str,
                                            party_col_name: str = 'participant',
                                            utter_col_name: str = 'utterance') -> Utterances:
        """Get utterances from a CSV file."""
        # Load the CSV file to a data frame.
        df = pd.read_csv(file_name)

        # Select 'utterance' values based on the participant mask
        mask = df[party_col_name] == party
        utterances_list = df.loc[mask, utter_col_name].tolist()

        utterances = Utterances(source=file_name, utterances=utterances_list)

        return utterances

    def extract_entities(self, file_path: str, output_dir: str):
        entities_by_type = defaultdict(set)
        entities_by_source = defaultdict(set)

        with codecs.open(file_path, 'r', 'utf-8') as file:
            for line in file:
                cols = line.split(',')
                if len(cols) != 3:
                    raise Exception(f"Invalid CSV transcript line {line}")

                source = cols[0]
                participant = cols[1]
                utterance = cols[2]

                doc = self.spacy_nlp(utterance)
                for ent in doc.ents:
                    # Add the entity text to the set corresponding to its type
                    entities_by_type[ent.label_].add(ent.text)
                    entities_by_source[ent.text].add(source)

        # Create a directory to store the entity files
        os.makedirs(output_dir, exist_ok=True)

        # Create and write to files for each entity type
        for entity_type, examples in entities_by_type.items():
            entity_name = entity_type.lower().replace(' ', '')
            entity_dir_path = os.path.join(output_dir, entity_name)
            os.makedirs(entity_dir_path, exist_ok=True)

            values_file_name = f"{entity_dir_path}/value_sources.csv"
            with codecs.open(values_file_name, 'w', 'utf-8') as f:
                f.write('source,value\n')
                for example in examples:
                    for source in entities_by_source[example]:
                        f.write(source+','+example + '\n')

            unique_file_name = f"{entity_dir_path}/unique_values.txt"
            with codecs.open(unique_file_name, 'w', 'utf-8') as f:
                for example in examples:
                    f.write(example+'\n')

    def cluster_and_name_utterances(self,
                                    *,
                                    output_dir: str,
                                    session_id: str = None,
                                    csv_file: str = None,
                                    csv_col: str = None,
                                    utterances: list[str] = None,
                                    min_cluster_size=40,
                                    min_samples=5,
                                    epsilon=0.2,
                                    ) -> None:
        """
        Cluster the utterances into groups, use generative AI to name them, and then store the results in files.
        """
        if session_id is None:
            session_id = Utils.new_session_id()

        # Initialize the Cluster Client.
        cluster_client = ClusterWrap(
            openai=self.openai,
            spacy_nlp=self.spacy_nlp,
            embeddings_model='MiniLM',
        )

        # Get the utterances.
        if utterances is None:
            if csv_file is None:
                raise Exception(f"utterances or csv_file are required inputs")

            if csv_col is None:
                csv_col = 'utterance'

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
        log.info(f"{session_id} | gen_agent_transcripts | Silhouette Score: {silhouette_avg:.2f}")

        clustered_sentences = cluster_client.get_clustered_sentences(utterances, cluster)

        # new_labels = cluster_client.get_new_cluster_labels(
        #     session_id=session_id,
        #     clustered_sentences=clustered_sentences,
        #     output_dir=output_dir,
        # )

        new_definitions = cluster_client.get_new_cluster_definitions(
            session_id=session_id,
            clustered_sentences=clustered_sentences,
            output_dir=output_dir,
        )

        new_labels = [d['name'] for d in new_definitions]

        cluster_client.visualize_clusters(umap_embeddings, labels, new_labels)

    @staticmethod
    def visualize_intent_bars(directory: str, utterance_counts: Counter = None):
        file_names = []
        line_counts = []

        title_key = 'Unique Utterances'

        for intent_dir in os.listdir(directory):
            intent_file_name = intent_dir+'.txt'
            intent_file_path = os.path.join(directory, intent_dir, intent_file_name)
            readme_file_path = os.path.join(directory, intent_dir, 'readme.txt')

            if os.path.isfile(intent_file_path):
                if intent_dir == "-1_noise":
                    continue

                description = ''
                if os.path.isfile(readme_file_path):
                    with codecs.open(readme_file_path, 'r', 'utf-8') as rf:
                        readme_value = rf.read()
                        description = '\n('+readme_value+')'

                normed_file = re.sub("[0-9]+_(.*?)", "\\1", intent_dir)
                normed_file = normed_file + description
                file_names.append(normed_file)

                if utterance_counts is not None:
                    title_key = 'Utterance Volume'
                    by_utterance_count = 0
                    with codecs.open(intent_file_path, 'r', 'utf-8') as open_file:
                        for line in open_file:
                            line_trim = line.strip()
                            if line_trim in utterance_counts:
                                final_count = utterance_counts.get(line_trim)
                                # print(f'{line_trim} has {final_count} instances')
                                by_utterance_count = by_utterance_count + final_count
                            else:
                                # print(f'{line_trim} not previously recorded')
                                by_utterance_count = by_utterance_count + 1
                    by_line_count = count_lines(intent_file_path)

                    # print(f'intent {normed_file} - by_line_count: {by_line_count}, by_utterance_count: {by_utterance_count}')

                    line_counts.append(by_utterance_count)
                else:
                    by_line_count = count_lines(intent_file_path)
                    line_counts.append(by_line_count)

        # Sort the files by line count in descending order
        sorted_data = sorted(zip(line_counts, file_names), reverse=True)
        line_counts_sorted, file_names_sorted = zip(*sorted_data)

        # Adjust the figure size to accommodate all entries
        plt.figure(figsize=(10, len(file_names_sorted) * 0.5))

        plt.barh(file_names_sorted, line_counts_sorted)
        plt.xlabel('Utterance Count')
        plt.title(f'Number of {title_key} per-Cluster (Sorted)')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
        plt.tight_layout()  # Adjust layout to fit all labels
        plt.show()

    @staticmethod
    def visualize_intent_pie(directory, utterance_counts: Counter = None):
        file_names = []
        line_counts = []

        title_key = 'Unique Utterances'

        for intent_dir in os.listdir(directory):
            intent_file_name = intent_dir+'.txt'
            intent_file_path = os.path.join(directory, intent_dir, intent_file_name)
            if os.path.isfile(intent_file_path):
                normed_file = re.sub("[0-9]+_(.*?)", "\\1", intent_dir)
                file_names.append(normed_file)

                if utterance_counts is not None:
                    title_key = 'Utterance Volume'
                    by_utterance_count = 0
                    with codecs.open(intent_file_path, 'r', 'utf-8') as open_file:
                        for line in open_file:
                            line_trim = line.strip()
                            if line_trim in utterance_counts:
                                final_count = utterance_counts.get(line_trim)
                                # print(f'{line_trim} has {final_count} instances')
                                by_utterance_count = by_utterance_count + final_count
                            else:
                                # print(f'{line_trim} not previously recorded')
                                by_utterance_count = by_utterance_count + 1
                    # by_line_count = count_lines(intent_file_path)
                    # print(f'intent {normed_file} - by_lines: {by_line_count}, by_utterances: {by_utterance_count}')

                    line_counts.append(by_utterance_count)
                else:
                    by_line_count = count_lines(intent_file_path)
                    line_counts.append(by_line_count)

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

        plt.title(f'Distribution of {title_key} by Cluster (<1% as Other, Sorted by Size)')
        plt.show()

    @staticmethod
    def visualize_entity_bars(directory: str):
        file_names = []
        line_counts = []

        for entity_dir in os.listdir(directory):
            entity_file_name = 'value_sources.csv'
            entity_file_path = os.path.join(directory, entity_dir, entity_file_name)
            if os.path.isfile(entity_file_path):
                entity_explain = spacy.explain(entity_dir.upper())

                file_names.append(entity_dir+'\n('+entity_explain+')')
                line_counts.append(count_lines(entity_file_path))

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
        for entity_dir in os.listdir(directory):
            entity_file_name = 'value_sources.csv'
            entity_file_path = os.path.join(directory, entity_dir, entity_file_name)
            if os.path.isfile(entity_file_path):
                file_names.append(entity_dir)
                line_counts.append(count_lines(entity_file_path))

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

    @staticmethod
    def get_intent_names(directory: str) -> list[str]:
        file_names = []

        for file in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, file)):
                normed_file = re.sub("[0-9]+_(.*?)", "\\1", file)
                file_names.append(normed_file)

        return file_names
