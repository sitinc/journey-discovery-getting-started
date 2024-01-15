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

import uuid
import os
import sys
import shutil
import codecs


class Utils:
    """Module utility class."""

    def __init__(self):
        """
        Construct a new instance.
        """
        raise Exception(f'Cannot instantiate this class.')

    @staticmethod
    def new_session_id():
        return str(uuid.uuid4())

    @staticmethod
    def progress_bar(progress: int, total: int, description: str = ''):
        """
        Displays or updates a console progress bar.

        :param progress: Current progress (should not exceed 'total').
        :param total: Total steps of the progress bar.
        :param description: The description of the progression activity.
        """
        bar_length = 40  # Modify this to change the length of the progress bar
        progress_length = int(round(bar_length * progress / float(total)))

        percent = round(100.0 * progress / float(total), 1)
        bar = '#' * progress_length + '-' * (bar_length - progress_length)

        if 'ipykernel' in sys.modules:  # Jupyter Notebook environment
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(f'[{bar}] {percent}% {description}')
        else:  # Console environment
            sys.stdout.write(f'\r[{bar}] {percent}% {description}')
            sys.stdout.flush()

    @staticmethod
    def copy_dir(*,
                 src_dir: str = None,
                 dst_dir: str = None,
                 incl_progress: bool = False,
                 ):
        """
        Copies files from one directory to another.

        :param src_dir: The source directory path.
        :param dst_dir: The destination directory path.
        :param incl_progress: The flag indicating to include a progress bar.
        """
        if src_dir is None or not os.path.isdir(src_dir):
            raise Exception('src_dir is required')

        if dst_dir is None or not os.path.isdir(dst_dir):
            raise Exception('dst_dir is required')

        with os.scandir(src_dir) as entries:
            file_count = sum(1 for entry in entries if entry.is_file())

        with os.scandir(src_dir) as entries:
            dir_progress = 0
            dir_progress_total = file_count
            for entry in entries:
                dir_progress = dir_progress + 1
                if incl_progress:
                    Utils.progress_bar(dir_progress, dir_progress_total, f'Copying file {entry.name}')
                if entry.is_file():
                    source_file = os.path.join(src_dir, entry.name)
                    destination_file = os.path.join(dst_dir, entry.name)
                    shutil.copy2(source_file, destination_file)

    @staticmethod
    def count_file_lines(file_path: str) -> int:
        """Utility function to count the number of lines in a file."""
        with codecs.open(file_path, 'r', 'utf-8') as file:
            return sum(1 for _ in file)
