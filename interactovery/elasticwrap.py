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

from interactovery.utils import Utils
from elasticsearch import Elasticsearch

import logging

log = logging.getLogger('elasticLogger')


class ElasticWrap:
    """ElasticSearch API Wrapper class."""
    def __init__(self, cloud_id: str = None, api_key: str = None):
        """
        Construct a new instance.
        :param cloud_id: The elastic cloud ID.
        :param api_key: The elastic cloud API key.
        """
        if cloud_id is None:
            raise Exception("cloud_id is required")
        if api_key is None:
            raise Exception("api_key is required")

        self.cloud_id = cloud_id
        self.api_key = api_key
        self.client = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key
        )

    def get_cluster_info(self):
        """
        Get the cluster info.
        """
        return self.client.info()

    def idx_tmpl_exists(self, template_name: str):
        """
        Check if an index template exists by name.
        :param template_name: The index template name.
        :return: True if the index template exists; otherwise, False.
        """
        return self.client.indices.exists_template(name=template_name)

    def idx_tmpl_add(self, template_name: str, index_template):
        """
        Add an index template.
        :param template_name: The index template name.
        :param index_template: The index template.
        :return: The API response object.
        """
        return self.client.indices.put_index_template(name=template_name, body=index_template)

    def idx_tmpl_add_not_exist(self, template_name: str, index_template):
        """
        Add an index template if it doesn't exist.
        :param template_name: The index template name.
        :param index_template: The index template.
        :return: The API response object, or None if it already exists.
        """
        if not self.idx_tmpl_exists(template_name):
            return self.idx_tmpl_add(template_name, index_template)
        return None
