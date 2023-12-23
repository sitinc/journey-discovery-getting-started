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
import os

from tsdiscovery.openaiwrap import OpenAiWrap, CreateCompletions, CreateEmbeddings

import logging
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.metrics import silhouette_score
import codecs

# For visualization
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Initialize the logger.
log = logging.getLogger(__name__)

sys_prompt_name_cluster = """You are helping me identify clusters based on utterances that are semantically similar."""
usr_prompt_name_cluster = """Below is a list of utterances.  Suggest an intent name that best fits the semantic 
similarity between these utterances.  Output the result as three words maximum in camel case with no spaces or special 
characters.\n\n"""


class ClusterWrap:
    """Implement the main Clustering AI Wrapper class."""
    def __init__(self,
                 *,
                 openai: OpenAiWrap,
                 embeddings_model: str):
        """Construct a new instance."""
        self.openai = openai
        self.embeddings_model = embeddings_model

    @staticmethod
    def reduce_dimensionality(*,
                              embeddings,
                              n_neighbors=15,
                              n_components=5,
                              metric='cosine',
                              ):
        """Reduce dimensionality."""
        umap_embeddings = umap.UMAP(n_neighbors=n_neighbors,
                                    n_components=n_components,
                                    metric=metric).fit_transform(embeddings)
        return umap_embeddings

    # Creates the embeddings for a set of utterances.
    def get_embeddings(self,
                       *,
                       session_id: str = None,
                       utterances: list[str]):
        """Get the embeddings."""
        if self.embeddings_model == 'MiniLM':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(utterances, show_progress_bar=True)
            return embeddings
        elif self.embeddings_model == 'OpenAI':
            cmd = CreateEmbeddings(
                session_id=session_id,
                utterances=utterances,
            )
            embeddings = self.openai.execute(cmd).result
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
        """Get the silhouette score for a set of clustered embeddings."""
        silhouette_avg = silhouette_score(embeddings, cluster.labels_)
        return silhouette_avg

    @staticmethod
    def get_clustered_sentences(utterances, cluster):
        """Get the clustered sentences."""
        clustered_sentences = {label: [] for label in set(cluster.labels_)}
        for sentence, cluster_label in zip(utterances, cluster.labels_):
            clustered_sentences[cluster_label].append(sentence)
        return clustered_sentences

    def get_new_cluster_labels(self,
                               *,
                               clustered_sentences,
                               output_dir: str,
                               min_utterances: int = 50,
                               ):
        """Name a set of sentences clusters."""
        new_labels = []

        os.makedirs(output_dir, exist_ok=True)

        # Display clusters
        for i, cluster_entries in clustered_sentences.items():
            utterances_text = ''
            all_utterances_text = ''

            # Only use the first set of entries to avoid too much API data transfer.
            for utterance in cluster_entries[:min_utterances]:
                utterances_text = f'{utterances_text}\n - {utterance}'

            # Use all utterances when outputting to a file.
            first_itr = 1
            for utterance in cluster_entries:
                if first_itr == 1:
                    all_utterances_text = f'{utterance}'
                    first_itr = 0
                else:
                    all_utterances_text = f'{all_utterances_text}\n{utterance}'

            if i == -1:
                with codecs.open(f'{output_dir}/-1_noise.txt', 'w', 'utf-8') as f:
                    f.write(all_utterances_text)
                continue

            # Generate and save the new clusters name.
            new_cluster_name = self.get_cluster_name(utterances=utterances_text)

            new_cluster_name_strip = new_cluster_name.strip()
            new_labels.append(new_cluster_name_strip)

            # print(f"\n\nCluster {new_cluster_name}:")
            # print(utterances_text)
            # print("")

            with codecs.open(f'{output_dir}/{i}_{new_cluster_name_strip}.txt', 'w', 'utf-8') as f:
                f.write(all_utterances_text)

        return new_labels

    def get_cluster_name(self,
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
        user_prompt = usr_prompt_name_cluster + utterances

        cmd = CreateCompletions(
            session_id=session_id,
            model=model,
            sys_prompt=sys_prompt_name_cluster,
            user_prompt=user_prompt,
        )
        return self.openai.execute(cmd).result

    @staticmethod
    def visualize_clusters(embeddings, labels, new_labels) -> None:
        """
        Visualize the named clusters on a graph.
        :param embeddings: The embeddings
        :param labels: The cluster labels
        :param new_labels: The generated cluster labels
        """
        # Visualize the clusters.
        tsne = TSNE(n_components=2, random_state=42)
        proj_2d = tsne.fit_transform(embeddings)

        # Plotting
        plt.figure(figsize=(20, 16))
        plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=labels, cmap='Spectral', s=50, alpha=0.7)

        # Calculate the centroid of each clusters
        for i in np.unique(labels):
            if i == -1:
                # Skip noise if necessary
                continue
            mask = labels == i
            new_label = new_labels[i]
            centroid = np.mean(proj_2d[mask], axis=0)
            plt.text(centroid[0], centroid[1], new_label, fontdict={'weight': 'bold', 'size': 10})

        plt.title('Clusters of Named Transcript Utterances (2D t-SNE Projection)', fontsize=15)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.colorbar(label='Cluster')
        plt.show()
