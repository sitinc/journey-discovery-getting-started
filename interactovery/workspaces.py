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

from interactovery import Utils


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
    Root container for interactional discovery resources.
    """
    def __init__(self,
                 *,
                 name: str = None,
                 work_dir: str = None,
                 ):
        """
        Construct a new instance.

        :param name: The name of the workspace.
        :param work_dir: The directory of the workspace.
        """
        if name is None:
            raise Exception('name is required')

        if work_dir is None:
            raise Exception('work_dir is required')

        self.name = name
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
                f", work_dir={self.work_dir!r}" +
                f", embeddings_path={self.embeddings_path!r}" +
                f", transcripts_path={self.transcripts_path!r}" +
                f", ts_combined_path={self.ts_combined_path!r}" +
                f", clusters_path={self.clusters_path!r}" +
                f", intents_path={self.intents_path!r}" +
                f", intent_groupings_path={self.intent_groupings_path!r}" +
                f", entities_path={self.entities_path!r}" +
                ")")

    def init(self):
        """
        Initialize the workspace.
        """
        for directory in self.workspace_dirs:
            os.makedirs(directory, exist_ok=True)


class WorkspaceManager:
    """
    Management of workspaces and the workspaces root.
    """
    def __init__(self,
                 *,
                 root_dir: str = None,
                 ):
        """
        Construct a new instance.

        :param root_dir: The root directory for all workspaces.
        """
        if root_dir is None:
            root_dir = os.path.join(os.getcwd(), DIRNAME_DEF_WORKSPACES)

        self.root_dir = root_dir

    def __str__(self):
        return f"WorkspaceManager(root_dir={self.root_dir})"

    def __repr__(self):
        return f"WorkspaceManager(root_dir={self.root_dir!r})"

    def create(self,
               *,
               name: str) -> Workspace:
        """
        Create a new workspace and return it.

        :param name: The workspace name.
        :return: The new workspace instance.
        """
        work_dir = os.path.join(self.root_dir, name)
        workspace = Workspace(
            name=name,
            work_dir=work_dir,
        )
        workspace.init()
        return workspace

    def copy(self,
             *,
             src_name: str = None,
             dst_name: str = None,
             incl_ts: bool = False,
             incl_ts_combined: bool = False,
             ) -> Workspace:
        """
        Copy a workspace.

        :param src_name: The source workspace name.
        :param dst_name: The destination workspace name.
        :param incl_ts: The flag indicating to copy the transcript directory.
        :param incl_ts_combined: The flag indicating to copy the combined transcript directory.
        :return: The new workspace instance.
        """
        if src_name is None:
            raise Exception('src_name is required')

        if dst_name is None:
            raise Exception('dst_name is required')

        src_work_dir = os.path.join(self.root_dir, src_name)
        if not os.path.isdir(src_work_dir):
            raise Exception(f'workspace {src_name} does not exist')
        src_workspace = Workspace(
            name=src_name,
            work_dir=src_work_dir,
        )
        src_workspace.init()

        dst_work_dir = os.path.join(self.root_dir, dst_name)
        dst_workspace = Workspace(
            name=dst_name,
            work_dir=dst_work_dir,
        )
        dst_workspace.init()

        if incl_ts:
            Utils.copy_dir(
                src_dir=src_workspace.transcripts_path,
                dst_dir=dst_workspace.transcripts_path,
            )

        if incl_ts_combined:
            Utils.copy_dir(
                src_dir=src_workspace.ts_combined_path,
                dst_dir=dst_workspace.ts_combined_path,
            )

        return dst_workspace
