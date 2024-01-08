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

import sys
from IPython.display import display, clear_output


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
    def progress_bar(progress: int, total: int):
        """
        Displays or updates a console progress bar.

        :param progress: Current progress (should not exceed 'total').
        :param total: Total steps of the progress bar.
        """
        bar_length = 40  # Modify this to change the length of the progress bar
        progress_length = int(round(bar_length * progress / float(total)))

        percent = round(100.0 * progress / float(total), 1)
        bar = '#' * progress_length + '-' * (bar_length - progress_length)

        if 'ipykernel' in sys.modules:  # Jupyter Notebook environment
            clear_output(wait=True)
            display(f'[{bar}] {percent}%')
        else:  # Console environment
            sys.stdout.write(f'\r[{bar}] {percent}%')
            sys.stdout.flush()
