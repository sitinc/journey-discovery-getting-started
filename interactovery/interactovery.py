# MIT License
#
# Copyright (c) 2024, Justin Randall, Smart Interactive Transformations Inc.
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

from interactovery import Utils, OpenAiWrap, ClusterWrap, VizWrap, MetricChart, Workspace
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import os
import re
import spacy
import codecs
import logging

# Initialize the logger.
log = logging.getLogger('interactoveryLogger')

DEF_ENTITY_VALUE_SOURCES = 'value_sources.csv'
DEF_ENTITY_UNIQUE_VALUES = 'unique_values.txt'

DEF_CLUSTERS_FILENAME = 'hdbscan_clusters.pkl'
DEF_CLUSTERS_DIRNAME = 'clusters'

DEF_CLUSTER_DEFS_FILENAME = 'cluster_definitions.pkl'

DEF_EMBEDDINGS_FILENAME = 'embeddings.pkl'
DEF_EMBEDDINGS_DIRNAME = 'embeddings'

DEF_REDUX_EMBEDDINGS_FILENAME = 'embeddings-reduced.pkl'


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


class Interactovery:
    """
    Interactovery client interface.
    """
    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 cluster: ClusterWrap,
                 spacy_nlp: spacy.language,
                 ):
        self.openai = openai
        self.cluster = cluster
        self.spacy_nlp = spacy_nlp

    def extract_entities(self,
                         *,
                         df: pd.DataFrame,
                         workspace: Workspace):
        entities_by_type = defaultdict(set)
        entities_by_source = defaultdict(set)

        entity_progress = 0
        entity_progress_total = df.shape[0]

        for row in df.itertuples():
            entity_progress = entity_progress + 1
            Utils.progress_bar(entity_progress, entity_progress_total, 'Extracting known entities')

            source = getattr(row, 'source')
            utterance = getattr(row, 'utterance')

            doc = self.spacy_nlp(utterance)
            for ent in doc.ents:
                # Add the entity text to the set corresponding to its type
                entities_by_type[ent.label_].add(ent.text)
                entities_by_source[ent.text].add(source)

        # Create and write to files for each entity type
        for entity_type, examples in entities_by_type.items():
            entity_name: str = entity_type.lower().replace(' ', '')
            entity_dir_path = os.path.join(workspace.entities_path, entity_name)
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
                    by_unique_count = Utils.count_file_lines(intent_file_path)
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
                metric_counts.append(Utils.count_file_lines(entity_file_path))

        metric_chart = MetricChart(
            title=title_key,
            metrics=metric_names,
            counts=metric_counts,
        )
        return metric_chart

    @staticmethod
    def detail_entity_types(directory):
        # Reading files and counting lines
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                scrub_file = re.sub("\\.txt", "", file)
                print(f"{scrub_file} - {spacy.explain(scrub_file.upper())}")

    def get_embeddings(self,
                       *,
                       session_id: str,
                       utterances: list[str],
                       ):
        """
        Get or create embeddings for a list of utterances.
        :param session_id: The session ID.
        :param utterances: The list of utterances.
        :return: the embeddings.
        """
        log.info(f"{session_id} | get_embeddings | Creating embeddings from utterances")
        embeddings = self.cluster.get_embeddings(utterances=utterances)
        return embeddings

    def reduce_embeddings(self,
                          *,
                          session_id: str,
                          embeddings,
                          ):
        """
        Reduce the dimensionality of provided embeddings with UMAP.
        :param session_id: The session ID.
        :param embeddings: The embeddings.
        :return: The reduced embeddings.
        """
        log.info(f"{session_id} | reduce_embeddings | Reducing dimensionality with UMAP")
        umap_embeddings = self.cluster.reduce_dimensionality(embeddings=embeddings)
        return umap_embeddings

    def cluster_scan(self,
                     *,
                     session_id: str,
                     embeddings,
                     min_cluster_size=40,
                     min_samples=5,
                     epsilon=0.2,
                     ):
        log.info(f"{session_id} | cluster_scan | Predicting cluster labels")
        clusters = self.cluster.hdbscan(
            embeddings=embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon,
        )
        return clusters

    def define_clusters(self,
                        *,
                        session_id: str,
                        workspace: Workspace,
                        clustered_sentences,
                        ):
        log.info(f"{session_id} | define_clusters | Generating cluster definitions from LLMs")
        new_definitions = self.cluster.get_new_cluster_definitions(
            session_id=session_id,
            clustered_sentences=clustered_sentences,
            output_dir=workspace.intents_dir_path,
        )
        return new_definitions

    def cluster_and_name_utterances(self,
                                    *,
                                    session_id: str = None,
                                    workspace: Workspace,
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

        # Get the embeddings.
        embeddings = self.get_embeddings(
            session_id=session_id,
            utterances=utterances,
        )

        # Reduce dimensionality.
        umap_embeddings = self.reduce_embeddings(
            session_id=session_id,
            embeddings=embeddings,
        )

        # Predict cluster labels.
        clusters = self.cluster_scan(
            session_id=session_id,
            embeddings=umap_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon,
        )

        labels = clusters.labels_

        # Get silhouette score.
        silhouette_avg = self.cluster.get_silhouette(umap_embeddings, clusters)
        log.info(f"{session_id} | cluster_and_name_utterances | Silhouette Score: {silhouette_avg:.2f}")

        clustered_sentences = self.cluster.get_clustered_sentences(utterances, clusters)

        new_definitions = self.define_clusters(
            session_id=session_id,
            workspace=workspace,
            clustered_sentences=clustered_sentences,
        )

        intent_labels = np.array([new_definitions[key].name for key in labels])

        log.info(f"{session_id} | cluster_and_name_utterances | Visualizing clusters")

        VizWrap.show_cluster_scatter(
            embeddings=umap_embeddings,
            labels=labels,
            intent_labels=intent_labels,
            silhouette_avg=silhouette_avg,
        )

    @staticmethod
    def save_raw_clusters(*,
                          workspace: Workspace,
                          clustered_sentences,
                          ):
        """
        Name a set of sentences clusters.
        :param workspace: The workspace.
        :param clustered_sentences: The sentence clusters
        """
        cluster_progress = 0
        cluster_progress_total = len(clustered_sentences.items())

        # Display clusters
        for i, cluster_entries in clustered_sentences.items():
            cluster_progress = cluster_progress + 1
            Utils.progress_bar(cluster_progress, cluster_progress_total, 'Writing raw labels for clusters')

            all_utterances_text = '\n'.join(cluster_entries)
            final_output_dir = f'{workspace.cluster_raw_dir_path}'

            with codecs.open(f'{final_output_dir}/{i}.txt', 'w', 'utf-8') as f:
                f.write(all_utterances_text)
        return True
