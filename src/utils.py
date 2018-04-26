from __future__ import print_function
from __future__ import division
from builtins import range
import csv
import codecs
import os
import json
from itertools import islice, chain
import pandas as pd


from fonduer.snorkel.utils import ProgressBar
from fonduer.snorkel.models import GoldLabel, GoldLabelKey, Document

from fonduer import HTMLPreprocessor

#from snorkel.contrib.fonduer import HTMLPreprocessor
#from snorkel.models import GoldLabel, GoldLabelKey, Document
#from snorkel.utils import ProgressBar

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
                    
                    
class MEMEXJsonPreprocessor(HTMLListPreprocessor):
    
    def __init__(self, path, file_list, encoding="utf-8", max_docs=float('inf')):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.file_list = file_list
        
    def _get_files(self,path_list):
        fpaths = [fl for fl in path_list]
        return fpaths
    
    def _can_read(self, fpath):
        return fpath.endswith('json')  
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        for fp in self._get_files(self.file_list):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(self.path, file_name):
                    yield doc, text
                    doc_count += 1
                    if doc_count >= self.max_docs:
                        return
                    
    def _lines_per_n(self, f, n):
        for line in f:
            yield ''.join(chain([line], islice(f, n - 1)))
        
    def _read_content_file(self, fl):
        json_lst = []
        #with codecs.open(fl, encoding=self.encoding) as f:
        with open(fl) as f:
            for chunk in self._lines_per_n(f, 6):
                jfile = json.loads(chunk)
                json_lst.append(jfile)
        json_pd = pd.DataFrame(json_lst).dropna()
        return json_pd
    
    def parse_file(self, fp, file_name):
        df = self._read_content_file(os.path.join(fp,file_name))
        for index, row in df.iterrows():
            name = row.url
            stable_id = self.get_stable_id(name)
            # getting rid of first and last quotes
            text = row.content[1:-1].encode(self.encoding)
            yield Document(name=name, stable_id=stable_id, text=str(text),
                               meta={'file_name' : file_name}), str(text)

class MEMEXJsonLPreprocessor(HTMLListPreprocessor):
    
    def __init__(self, path, file_list, encoding="utf-8", max_docs=float('inf'), lines_per_entry=6, verbose=False):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs
        self.file_list = file_list
        self.lines_per_entry = lines_per_entry
        self.verbose=verbose
        
    def _get_files(self,path_list):
        fpaths = [fl for fl in path_list]
        return fpaths
    
    def _can_read(self, fpath):
        return fpath.endswith('jsonl')  
    
    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.
        """
        doc_count = 0
        for file_name in self._get_files(self.file_list):
            if self._can_read(file_name):
                for doc, text in self.parse_file(file_name):
                    yield doc, text
                    doc_count += 1
                    if self.verbose:
                        print(f'Parsed {doc_count} docs...')
                    if doc_count >= self.max_docs:
                        return
                    
    def _lines_per_n(self, f, n):
        for line in f:
            yield ''.join(chain([line], islice(f, n - 1)))
        
    def _read_content_file(self, fl):
        json_lst = []
        #with codecs.open(fl, encoding=self.encoding) as f:
        with open(fl) as f:
            for chunk in self._lines_per_n(f, self.lines_per_entry):
                jfile = json.loads(chunk)
                json_lst.append(jfile)
        json_pd = pd.DataFrame(json_lst)
       # json_pd = pd.DataFrame(json_lst).dropna()
        return json_pd
    
    def parse_file(self, file_name):
        df = self._read_content_file(file_name)
        for index, row in df.iterrows():
            name = row.url
            stable_id = self.get_stable_id(name)
            text = row.raw_content[1:-1].encode(self.encoding)
            yield Document(name=name, stable_id=stable_id, text=str(text),
                               meta={'file_name' : file_name}), str(text)
    