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
import os
import os.path


DIRNAME_DEF_WORKSPACES = 'workspaces'

DIRNAME_TRANSCRIPTS = 'transcripts'
DIRNAME_TRANSCRIPTS_COMBINED = 'transcripts-combined'
DIRNAME_EMBEDDINGS = 'embeddings'
DIRNAME_CLUSTERS = 'clusters'
DIRNAME_INTENTS = 'intents'
DIRNAME_INTENT_GROUPINGS = 'intent-groupings'
DIRNAME_ENTITIES = 'entities'


class Workspace:
    """
    Root container for interactional discovery outputs.
    """
    def __init__(self,
                 *,
                 name: str = None,
                 root_dir: str = None,
                 ):
        """
        Construct a new instance.

        :param name: The name of the workspace.
        :param root_dir: The root directory for all workspaces.
        """

        if name is None:
            raise Exception('name is required')

        if root_dir is None:
            root_dir = os.path.join(os.getcwd(), DIRNAME_DEF_WORKSPACES)

        work_dir = os.path.join(root_dir, name)

        self.name = name
        self.root_dir = root_dir
        self.work_dir = work_dir

        embeddings_path = os.path.join(self.work_dir, DIRNAME_EMBEDDINGS)
        self.embeddings_path = embeddings_path

        transcripts_path = os.path.join(self.work_dir, DIRNAME_TRANSCRIPTS)
        self.transcripts_path = transcripts_path

        ts_combined_path = os.path.join(self.work_dir, DIRNAME_TRANSCRIPTS_COMBINED)
        self.ts_combined_path = ts_combined_path

        clusters_path = os.path.join(self.work_dir, DIRNAME_CLUSTERS)
        self.clusters_path = clusters_path

        intents_path = os.path.join(self.work_dir, DIRNAME_INTENTS)
        self.intents_path = intents_path

        intent_groupings_path = os.path.join(self.work_dir, DIRNAME_INTENT_GROUPINGS)
        self.intent_groupings_path = intent_groupings_path

        entities_path = os.path.join(self.work_dir, DIRNAME_ENTITIES)
        self.entities_path = entities_path

        self.workspace_dirs = [
            embeddings_path,
            transcripts_path,
            ts_combined_path,
            clusters_path,
            intents_path,
            intent_groupings_path,
            entities_path,
        ]

    def __str__(self):
        return (f"Workspace(name={self.name}" +
                f", root_dir={self.root_dir}" +
                f", work_dir={self.work_dir}" +
                f", embeddings_path={self.embeddings_path}" +
                f", transcripts_path={self.transcripts_path}" +
                f", ts_combined_path={self.ts_combined_path}" +
                f", clusters_path={self.clusters_path}" +
                f", intents_path={self.intents_path}" +
                f", intent_groupings_path={self.intent_groupings_path}" +
                f", entities_path={self.entities_path}" +
                ")")

    def __repr__(self):
        return (f"Workspace(name={self.name!r}" +
                f", root_dir={self.root_dir!r}" +
                f", work_dir={self.work_dir!r}" +
                f", embeddings_path={self.embeddings_path!r}" +
                f", transcripts_path={self.transcripts_path!r}" +
                f", ts_combined_path={self.ts_combined_path!r}" +
                f", clusters_path={self.clusters_path!r}" +
                f", intents_path={self.intents_path!r}" +
                f", intent_groupings_path={self.intent_groupings_path!r}" +
                f", entities_path={self.entities_path!r}" +
                ")")

    def create(self):
        for directory in self.workspace_dirs:
            os.makedirs(directory, exist_ok=True)
