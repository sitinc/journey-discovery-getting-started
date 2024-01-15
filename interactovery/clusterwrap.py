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

import os
import re

import json

from interactovery import Utils, OpenAiWrap, CreateCompletions

import logging
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.metrics import silhouette_score
import codecs
import spacy
import pandas as pd


# Initialize the logger.
log = logging.getLogger('clusterLogger')

sys_prompt_name_cluster = """You are helping me identify clusters based on utterances that are semantically similar."""
usr_prompt_name_cluster = """Below is a list of utterances.  Suggest an intent name that best fits the semantic 
similarity between these utterances.  Output the result as three words maximum in camel case with no spaces or special 
characters.  Don't use the word \"intent\" in the intent name.\n\n"""

sys_prompt_define_cluster = """You are helping me identify cluster names and meanings based on semantically similarity 
of utterances."""
usr_prompt_define_cluster = """Below is a list of utterances.  Suggest an intent name and a description that best fits 
the semantic similarity between these utterances.  Output the result as a JSON object with two parameters.  The first 
parameter, "intent", is a string representing the intent name.  The second parameter "description" is a single sentence 
  describing the meaning of the relationship between the utterances.  The "description" should be no more than 12 
  words.  The intent name should be three words maximum in camel case with no spaces or special characters.  Don't use 
  the word \"intent\" in the intent name.\n\n"""

sys_prompt_group_intents = """You are helping me group semantically similar intents_old together."""
usr_prompt_group_intents = """Below is a list of intent names that were generated based on utterances having semantic 
similarity.  Reply back with a JSON object with a property name representing each group, and it's value set to the 
array of grouped intent names.  Don't put the JSON object into a string and don't wrap it with ```json or backticks.
\n\n"""


class IntentDefinition:
    """
    Intent definition.
    """
    def __init__(self,
                 *,
                 label: str = None,
                 name: str = None,
                 brief: str = None,
                 description: str = None,
                 ):
        self.label = label
        self.name = name
        self.brief = brief
        self.description = description

    def __str__(self):
        return (f"IntentDefinition(label={self.label}" +
                f", name={self.name}" +
                f", brief={self.brief}" +
                f", description={self.description}" +
                ")")

    def __repr__(self):
        return (f"IntentDefinition(label={self.label!r}" +
                f", name={self.name!r}" +
                f", brief={self.brief!r}" +
                f", description={self.description!r}" +
                ")")


class ClusterWrap:
    """
    Implement the main Clustering AI Wrapper class.
    """
    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 spacy_nlp: spacy.language,
                 embeddings_model: str = 'MiniLM'):
        """
        Construct a new instance.
        :param openai: The OpenAI wrapper
        :param embeddings_model: The embedding model type name.  Currently only 'MiniLM' is support (and is default)
        """
        self.openai = openai
        self.spacy_nlp = spacy_nlp
        self.embeddings_model = embeddings_model

    def preprocess_text(self, text):
        """
        Preprocess the text.
        :param text: The text.
        :return: The preprocessed text.
        """
        doc = self.spacy_nlp(text)

        # Generate a list of tokens after preprocessing
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

        preprocessed_text = " ".join(tokens)
        return preprocessed_text

    @staticmethod
    def reduce_dimensionality(*,
                              embeddings,
                              n_neighbors=15,
                              n_components=5,
                              metric='cosine',
                              ):
        """
        Reduce dimensionality of the embeddings.
        :param embeddings: The embeddings
        :param n_neighbors: The
        :param n_components: The
        :param metric: The

        Parameters
        ----------
        n_neighbors: float (optional, default 15)
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller
            values result in more local data being preserved. In general
            values should be in the range 2 to 100.

        n_components: int (optional, default 2)
            The dimension of the space to embed into. This defaults to 2 to
            provide easy visualization, but can reasonably be set to any
            integer value in the range 2 to 100.

        metric: string or function (optional, default 'euclidean')
            The metric to use to compute distances in high dimensional space.
            If a string is passed it must match a valid predefined metric. If
            a general metric is required a function that takes two 1d arrays and
            returns a float can be provided. For performance purposes it is
            required that this be a numba jit'd function. Valid string metrics
            include:

            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * ll_dirichlet
            * hellinger
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule

            Metrics that take arguments (such as minkowski, mahalanobis etc.)
            can have arguments passed via the metric_kwds dictionary. At this
            time care must be taken and dictionary elements must be ordered
            appropriately; this will hopefully be fixed in the future.
        """
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components,
                                    metric=metric).fit_transform(embeddings)
        return umap_embeddings

    # Creates the embeddings for a set of utterances.
    def get_embeddings(self,
                       *,
                       session_id: str = None,
                       utterances: list[str]):
        """
        Get the embeddings.
        :param session_id: The session ID.
        :param utterances: The utterances.
        """

        # preprocessed_texts = [self.preprocess_text(text) for text in utterances]
        preprocessed_texts = utterances

        if self.embeddings_model == 'MiniLM':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(preprocessed_texts, show_progress_bar=True)
            return embeddings
        else:
            raise Exception(f'Unsupported embeddings model: {self.embeddings_model}')

    @staticmethod
    def hdbscan(*,
                embeddings,
                min_cluster_size=10,
                min_samples=5,
                epsilon=0.2,
                metric='euclidean',
                cluster_selection_method='eom'
                ):
        """Cluster the embeddings using the HDBSCAN algorithm."""
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=min_samples,
                                  cluster_selection_epsilon=epsilon,
                                  metric=metric,
                                  cluster_selection_method=cluster_selection_method).fit(embeddings)
        return cluster

    @staticmethod
    def get_silhouette(embeddings, cluster):
        """
        Get the silhouette score for a set of clustered embeddings.
        :param embeddings: The embeddings
        :param cluster: The clusters
        """
        silhouette_avg = silhouette_score(embeddings, cluster.labels_)
        return silhouette_avg

    @staticmethod
    def get_clustered_sentences(utterances, cluster):
        """
        Get the cluster of original utterances from the clustered embeddings.
        :param utterances: The original utterances.
        :param cluster: The clustered embeddings.
        :return: The clustered utterances.
        """
        clustered_sentences = {label: [] for label in set(cluster.labels_)}
        for sentence, cluster_label in zip(utterances, cluster.labels_):
            clustered_sentences[cluster_label].append(sentence)
        return clustered_sentences

    def get_new_cluster_definitions(self,
                                    *,
                                    session_id: str = None,
                                    clustered_sentences,
                                    output_dir: str,
                                    max_samples: int = 50,
                                    ) -> dict:
        """
        Name a set of sentences clusters.
        :param session_id: The session ID.
        :param clustered_sentences: The sentence clusters
        :param output_dir: The output directory
        :param max_samples: The number of sample utterances to include in the LLM cluster name call.
        """
        cluster_labels = []
        intent_definitions = []

        cluster_progress = 0
        cluster_progress_total = len(clustered_sentences.items())

        # Loop through label names and utterances.
        for i, cluster_entries in clustered_sentences.items():
            if session_id is None:
                session_id = Utils.new_session_id()

            cluster_progress = cluster_progress + 1
            Utils.progress_bar(cluster_progress, cluster_progress_total, 'Generating names for clusters')

            # Only use max samples of entries to avoid too much API data transfer.
            prefixed_utterances = [" - " + s for s in cluster_entries[:max_samples]]
            sample_utterances_text = '\n'.join(prefixed_utterances)

            # Use all utterances when outputting to a file.
            all_utterances_text = '\n'.join(cluster_entries)

            if i == -1:
                final_output_dir = f'{output_dir}/-1_noise'
                os.makedirs(final_output_dir, exist_ok=True)
                cluster_labels.append(i)
                intent_definitions.append(IntentDefinition(
                    label=i,
                    name='Noise',
                    description='Utterances not assigned to any cluster.'
                ))
                with codecs.open(f'{final_output_dir}/-1_noise.txt', 'w', 'utf-8') as f:
                    f.write(all_utterances_text)
                continue

            # Generate and save the new clusters name.
            log.debug(f"{session_id} | get_new_cluster_labels | Generating name for Cluster #{i}")
            new_cluster_definition_str = self.get_cluster_definition(utterances=sample_utterances_text)
            new_cluster_definition = json.loads(new_cluster_definition_str)

            log.debug(f"{session_id} | get_new_cluster_labels | Cluster #{i}: {new_cluster_definition}")

            new_cluster_name = new_cluster_definition['intent']
            new_cluster_descr = new_cluster_definition['description']

            new_cluster_name_strip = new_cluster_name.strip()

            cluster_labels.append(i)
            intent_definitions.append(IntentDefinition(
                label=i,
                name=new_cluster_name_strip,
                description=new_cluster_descr
            ))

            final_output_dir = f'{output_dir}/{i}_{new_cluster_name_strip}'
            os.makedirs(final_output_dir, exist_ok=True)

            with codecs.open(f'{final_output_dir}/readme.txt', 'w', 'utf-8') as f:
                f.write(new_cluster_descr)

            with codecs.open(f'{final_output_dir}/{i}_{new_cluster_name_strip}.txt', 'w', 'utf-8') as f:
                f.write(all_utterances_text)

        new_definitions = dict(zip(cluster_labels, intent_definitions))

        return new_definitions

    def get_cluster_definition(self,
                               *,
                               utterances: str,
                               model: str = "gpt-4-1106-preview",
                               session_id: str = None
                               ):
        """
        Generate an agent transcript.
        :param model: The OpenAI chat completion model.
        :param utterances: The utterances.
        :param session_id: The session ID.
        :return: the result transcript.
        """
        user_prompt = usr_prompt_define_cluster + utterances

        cmd = CreateCompletions(
            session_id=session_id,
            model=model,
            sys_prompt=sys_prompt_define_cluster,
            user_prompt=user_prompt,
        )
        result = self.openai.execute(cmd).result
        result = result.strip('```json')
        result = result.strip('`')
        return result

    def get_grouped_intent_names(self,
                                 *,
                                 session_id: str = None,
                                 intent_names: list[str],
                                 output_dir: str,
                                 model: str = "gpt-4-1106-preview",
                                 ):
        """
        Group similar intent names together.
        :param session_id: The session ID.
        :param intent_names: The intent names.
        :param output_dir: The output directory.
        :param model: The OpenAI chat completion model.
        """
        if session_id is None:
            session_id = Utils.new_session_id()

        intent_names_text = '\n'.join(intent_names)

        # Generate and save the new clusters name.
        session_id = Utils.new_session_id()
        log.info(f"{session_id} | get_grouped_intent_names | Getting groups for Cluster Names")

        user_prompt = usr_prompt_group_intents + intent_names_text

        cmd = CreateCompletions(
            session_id=session_id,
            model=model,
            sys_prompt=sys_prompt_group_intents,
            user_prompt=user_prompt,
        )
        cmd_result = self.openai.execute(cmd)
        result = cmd_result.result
        log.debug(f"{session_id} | get_grouped_intent_names | result: {result}")

        with codecs.open(f'{output_dir}/cluster_groupings.txt', 'w', 'utf-8') as f:
            f.write(result)

        # Convert the dictionary to a DataFrame
        max_len = max(len(v) for v in result.values())
        for key in result:
            result[key] += [''] * (max_len - len(result[key]))

        df = pd.DataFrame(result)

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(output_dir, 'intent_groupings_before.csv')
        df.to_csv(csv_file_path, index=False)
        return result

    @staticmethod
    def get_intent_names(directory: str, remove_dups: bool = False) -> list[str]:
        file_names = []

        files = os.listdir(directory)
        files.sort()

        for file in files:
            if os.path.isdir(os.path.join(directory, file)):
                if file == '-1_noise':
                    continue
                normed_file = re.sub("[0-9]+_(.*?)", "\\1", file)
                file_names.append(normed_file)

        if remove_dups:
            file_names_set = set(file_names)
            final_file_names = list(file_names_set)
            return final_file_names

        return file_names
