from __future__ import print_function
from __future__ import division
from builtins import range
import csv
import codecs
import os

from snorkel.utils import ProgressBar
from snorkel.models import GoldLabel, GoldLabelKey

from snorkel.contrib.fonduer import HTMLPreprocessor

class HTMLListPreprocessor(HTMLPreprocessor):
    
    def __init__(self, path, file_list, encoding="utf-8", max_docs=float('inf')):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.file_list = file_list
        
    def _get_files(self,path_list):
        fpaths = [os.path.join(self.path,fl) for fl in path_list]
        return fpaths
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        for fp in self._get_files(self.file_list):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(fp, file_name):
                    yield doc, text
                    doc_count += 1
                    if doc_count >= self.max_docs:
                        return
    
    
    