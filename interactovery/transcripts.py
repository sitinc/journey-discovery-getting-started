# MIT License
#
# Copyright (c) 2023, Justin Randall, Smart Interactive Transformations Inc.
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
import pickle
import pandas as pd
import codecs
import os
import re
import spacy
import logging
from collections import Counter

from interactovery.openaiwrap import OpenAiWrap, CreateCompletions
from interactovery.clusterwrap import ClusterWrap
from interactovery.vizwrap import MetricChart
from interactovery.utils import Utils

log = logging.getLogger('transcriptLogger')

sys_prompt_gen_transcript = """You are helping me generate example transcripts.  Do not reply back with anything other 
than the transcript content itself.  No headers or footers.  Only generate a single transcript example for each 
response.  Separate each turn with a blank line.  Each line should start with either "USER: " or "AGENT: "."""

DEF_TS_COMBINED_FILENAME = 'transcripts_combined.csv'

DEF_CLUSTERS_FILENAME = 'hdbscan_clusters.pkl'
DEF_CLUSTERS_DIRNAME = 'clusters'

DEF_CLUSTER_DEFS_FILENAME = 'cluster_definitions.pkl'

DEF_EMBEDDINGS_FILENAME = 'embeddings.pkl'
DEF_EMBEDDINGS_DIRNAME = 'embeddings'

DEF_REDUX_EMBEDDINGS_FILENAME = 'embeddings-reduced.pkl'

DEF_ENTITY_VALUE_SOURCES = 'value_sources.csv'
DEF_ENTITY_UNIQUE_VALUES = 'unique_values.txt'


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

        volume_utterance_count_map = Counter(utterances)
        self.volume_utterance_count_map = volume_utterance_count_map

        volume_utterance_counts_values = sum(volume_utterance_count_map.values())

        if volume_utterance_counts_values != utterance_count:
            raise Exception(f'Consistency failure: ({volume_utterance_counts_values} != {utterance_count})')

    def __str__(self):
        return (f"Utterances(source={self.source}" +
                f", utterances=..." +
                f", utterance_count={self.utterance_count}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count}" +
                f", volume_utterance_count_map=..." +
                ")")

    def __repr__(self):
        return (f"Utterances(source={self.source!r}" +
                f", utterances=..." +
                f", utterances_count={self.utterance_count!r}" +
                f", unique_utterances=..." +
                f", unique_utterance_count={self.unique_utterance_count!r}" +
                f", volume_utterance_count_map=..." +
                ")")


def count_lines(file_path: str) -> int:
    with codecs.open(file_path, 'r', 'utf-8') as file:
        return sum(1 for line in file)


def count_utterances(file_path: str, utterance_counts: Counter) -> int:
    by_utterance_count = 0
    with codecs.open(file_path, 'r', 'utf-8') as file:
        for line in file:
            line_trim = line.strip()
            if line_trim in utterance_counts:
                final_count = utterance_counts.get(line_trim)
                by_utterance_count = by_utterance_count + final_count
            else:
                by_utterance_count = by_utterance_count + 1
    return by_utterance_count


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
                              output_dir: str = "output",
                              offset: int = 0,
                              ) -> None:
        """
        Generate a series of agent transcripts and output them to files.
        :param user_prompt: The chat completion user prompt.
        :param session_id: The session ID.
        :param quantity: The number of transcripts to generate.
        :param model: The OpenAI chat completion model.  Default is "gpt-4-1106-preview"
        :param output_dir: The transcript file output directory.
        :param offset: The transcript file number offset.
        :return: transcripts will be output to file system.
        """
        if quantity > self.max_transcripts:
            raise Exception(f"max quantity is {self.max_transcripts} unless you set max_transcripts via constructor")

        if session_id is None:
            session_id = Utils.new_session_id()

        os.makedirs(output_dir, exist_ok=True)
        file_progress = 0
        file_progress_total = quantity

        for i in range(offset, offset+quantity):
            file_progress = file_progress + 1
            Utils.progress_bar(file_progress, file_progress_total, 'Generating transcripts')

            final_file_name = f'{output_dir}/transcript{i}.txt'

            if os.path.exists(final_file_name):
                continue

            log.debug(f"{session_id} | gen_agent_transcripts | Generating example transcript #{i}")
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
                                     in_dir: str = None,
                                     out_dir: str = None,
                                     out_file: str = DEF_TS_COMBINED_FILENAME) -> str | None:
        """
        Concatenate and process the transcripts
        :param in_dir: The input directory with transcripts.
        :param out_dir: The output directory for the combined CSV.
        :param out_file: The name of the combined transcripts CSV output file.
        :return: True if no errors occurred.
        """
        if in_dir is None:
            raise Exception('in_dir is required')

        if out_dir is None:
            raise Exception('out_dir is required')

        out_file_path = os.path.join(out_dir, out_file)

        if os.path.isfile(out_file_path):
            return out_file_path

        try:
            os.makedirs(out_dir, exist_ok=True)

            with codecs.open(out_file_path, 'w+', 'utf-8') as wf:
                wf.write("source,participant,utterance\n")

            ts_files = os.listdir(in_dir)

            file_progress = 0
            file_progress_total = len(ts_files)

            for ts_file in ts_files:
                file_progress = file_progress + 1
                Utils.progress_bar(file_progress, file_progress_total, 'Assembling transcripts to CSV file')
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

                            with codecs.open(out_file_path, 'a+', 'utf-8') as wf:
                                wf.write(csv_text)
                except UnicodeDecodeError as err:
                    log.error(f"Unicode decode error for file: {ts_file}: {err.reason}")
                    return None
        except Exception as err:
            log.error(f"Unhandled exception: {err}")
            return None

        return out_file_path

    @staticmethod
    def get_combined_utterances(*,
                                in_dir: str,
                                source: str | None = DEF_TS_COMBINED_FILENAME,
                                ) -> pd.DataFrame:

        path = os.path.join(in_dir, source)
        df = pd.read_csv(path)
        return df

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

    def extract_entities(self, df: pd.DataFrame, output_dir: str):
        entities_by_type = defaultdict(set)
        entities_by_source = defaultdict(set)

        entity_progress = 0
        entity_progress_total = df.shape[0]

        for row in df.itertuples():
            entity_progress = entity_progress + 1
            Utils.progress_bar(entity_progress, entity_progress_total, 'Extracting known entities')

            source = getattr(row, 'source')
            participant = getattr(row, 'participant')
            utterance = getattr(row, 'utterance')

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

            values_file_path = os.path.join(entity_dir_path, DEF_ENTITY_VALUE_SOURCES)
            with codecs.open(values_file_path, 'w', 'utf-8') as f:
                f.write('source,value\n')
                for example in examples:
                    for source in entities_by_source[example]:
                        f.write(source+','+example + '\n')

            unique_file_path = os.path.join(entity_dir_path, DEF_ENTITY_UNIQUE_VALUES)
            with codecs.open(unique_file_path, 'w', 'utf-8') as f:
                for example in examples:
                    f.write(example+'\n')

    def cluster_and_name_utterances(self,
                                    *,
                                    workspace_dir: str,
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

        # Create/Load/Store the embeddings.
        embeddings_file_name = DEF_EMBEDDINGS_FILENAME
        embeddings_dir = os.path.join(workspace_dir, DEF_EMBEDDINGS_DIRNAME)
        embeddings_file_path = os.path.join(embeddings_dir, embeddings_file_name)

        if os.path.isfile(embeddings_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading embeddings from {embeddings_file_name}")
            # embeddings = np.load(embeddings_file_path)
            with open(embeddings_file_path, 'rb') as file:
                embeddings = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Generating embeddings with 'all-MiniLM-L6-v2'")
            embeddings = cluster_client.get_embeddings(utterances=utterances)
            # np.save(embeddings_file_path, embeddings)
            with open(embeddings_file_path, 'wb') as file:
                pickle.dump(embeddings, file)

        # Reduce dimensionality.
        redux_file_name = DEF_REDUX_EMBEDDINGS_FILENAME
        redux_file_path = os.path.join(embeddings_dir, redux_file_name)

        if os.path.isfile(redux_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading reduced embeddings from {redux_file_name}")
            # umap_embeddings = np.load(redux_file_path)
            with open(redux_file_path, 'rb') as file:
                umap_embeddings = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Reducing dimensionality with UMAP")
            umap_embeddings = cluster_client.reduce_dimensionality(embeddings=embeddings)
            # np.save(redux_file_path, embeddings)
            with open(redux_file_path, 'wb') as file:
                pickle.dump(umap_embeddings, file)

        # Create/Load/Store the cluster results.
        clusters_file_name = DEF_CLUSTERS_FILENAME
        clusters_dir = os.path.join(workspace_dir, DEF_CLUSTERS_DIRNAME)
        clusters_file_path = os.path.join(clusters_dir, clusters_file_name)
        if os.path.isfile(clusters_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading clusters from {clusters_file_name}")
            with open(clusters_file_path, 'rb') as file:
                cluster = pickle.load(file)
        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Predicting cluster labels with HDBSCAN")
            cluster = cluster_client.hdbscan(
                embeddings=umap_embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                epsilon=epsilon,
            )
            with open(clusters_file_path, 'wb') as file:
                pickle.dump(cluster, file)

        labels = cluster.labels_

        # Get silhouette score.
        silhouette_avg = cluster_client.get_silhouette(umap_embeddings, cluster)
        log.info(f"{session_id} | cluster_and_name_utterances | Silhouette Score: {silhouette_avg:.2f}")

        clustered_sentences = cluster_client.get_clustered_sentences(utterances, cluster)

        definitions_file_name = DEF_CLUSTER_DEFS_FILENAME
        definitions_file_path = os.path.join(clusters_dir, definitions_file_name)
        if os.path.isfile(definitions_file_path):
            log.info(f"{session_id} | cluster_and_name_utterances | Loading definitions from {definitions_file_name}")
            with open(definitions_file_path, 'rb') as file:
                new_definitions = pickle.load(file)

        else:
            log.info(f"{session_id} | cluster_and_name_utterances | Generating definitions from LLMs")
            new_definitions = cluster_client.get_new_cluster_definitions(
                session_id=session_id,
                clustered_sentences=clustered_sentences,
                output_dir=output_dir,
            )
            with open(definitions_file_path, 'wb') as file:
                pickle.dump(new_definitions, file)

        new_labels = [d['name'] for d in new_definitions]

        log.info(f"{session_id} | cluster_and_name_utterances | Visualizing clusters")

        cluster_client.visualize_clusters(umap_embeddings, labels, new_labels, silhouette_avg)

    @staticmethod
    def get_intent_utterance_counts(
            *,
            directory: str,
            utterance_volumes: Counter = None,
            incl_descr: bool = True,
            incl_noise: bool = False,
            descr_sep: str = '\n',
    ) -> MetricChart:
        metric_names = []
        metric_counts = []

        title_key = 'Unique Utterances'

        for intent_dir in os.listdir(directory):
            intent_file_name = intent_dir + '.txt'
            intent_file_path = os.path.join(directory, intent_dir, intent_file_name)
            readme_file_path = os.path.join(directory, intent_dir, 'readme.txt')

            if os.path.isfile(intent_file_path):
                if not incl_noise and intent_dir == "-1_noise":
                    continue

                normed_file = re.sub("[0-9]+_(.*?)", "\\1", intent_dir)
                if incl_descr:
                    description = ''
                    if os.path.isfile(readme_file_path):
                        with codecs.open(readme_file_path, 'r', 'utf-8') as rf:
                            readme_value = rf.read()
                            description = descr_sep+'(' + readme_value + ')'
                    normed_file = normed_file + description
                metric_names.append(normed_file)

                if utterance_volumes is not None:
                    title_key = 'Utterance Volume'
                    by_volume_count = count_utterances(intent_file_path, utterance_volumes)
                    metric_counts.append(by_volume_count)
                else:
                    by_unique_count = count_lines(intent_file_path)
                    metric_counts.append(by_unique_count)

        metric_chart = MetricChart(
            title=title_key,
            metrics=metric_names,
            counts=metric_counts,
        )
        return metric_chart

    @staticmethod
    def get_entity_value_counts(
            *,
            directory: str,
            incl_descr: bool = True,
            descr_sep: str = '\n'
    ) -> MetricChart:
        metric_names = []
        metric_counts = []

        title_key = 'Value Volume'

        for entity_dir in os.listdir(directory):
            entity_file_name = DEF_ENTITY_VALUE_SOURCES
            entity_file_path = os.path.join(directory, entity_dir, entity_file_name)
            if os.path.isfile(entity_file_path):
                entity_label = entity_dir
                if incl_descr:
                    entity_explain = spacy.explain(entity_dir.upper())
                    entity_label = entity_label + descr_sep + '(' + entity_explain + ')'

                metric_names.append(entity_label)
                metric_counts.append(count_lines(entity_file_path))

        metric_chart = MetricChart(
            title=title_key,
            metrics=metric_names,
            counts=metric_counts,
        )
        return metric_chart

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

        files = os.listdir(directory)
        files.sort()

        for file in files:
            if os.path.isdir(os.path.join(directory, file)):
                if file == '-1_noise':
                    continue
                normed_file = re.sub("[0-9]+_(.*?)", "\\1", file)
                file_names.append(normed_file)

        return file_names
