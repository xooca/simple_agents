# agentic_orchestrator/data_ingestion/sources.py
import os
from typing import List, Any, Optional, Dict
import pandas as pd
import pickle
import json

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    UnstructuredFileLoader,
)

from .base import DataSource
from ..exceptions import DataSourceError


class PDFSource(DataSource):
    """Concrete data source for PDF files."""
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata.setdefault("source_file_basename", os.path.basename(self.file_path))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load PDF from {self.file_path}: {e}")


class CSVSource(DataSource):
    """Concrete data source for CSV files."""
    def __init__(self, file_path: str, source_column: Optional[str] = None, csv_args: Optional[Dict[str, Any]] = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at {file_path}")
        self.file_path = file_path
        self.source_column = source_column
        self.csv_args = csv_args or {}

    def load(self) -> List[Document]:
        try:
            loader = CSVLoader(
                file_path=self.file_path,
                source_column=self.source_column,
                csv_args=self.csv_args
            )
            documents = loader.load()
            for doc in documents:
                doc.metadata.setdefault("source_file_basename", os.path.basename(self.file_path))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load CSV from {self.file_path}: {e}")


class WordSource(DataSource):
    """Concrete data source for Word (DOCX) files."""
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Word file not found at {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            loader = Docx2txtLoader(self.file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata.setdefault("source_file_basename", os.path.basename(self.file_path))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load Word document from {self.file_path}: {e}")


class JSONSource(DataSource):
    """
    Concrete data source for JSON files.
    Uses JSONLoader, which requires a jq schema to extract text.
    Example jq_schema: '.content' for a flat JSON with a "content" field.
    For nested data: '.posts[].text' to extract "text" from each item in "posts" array.
    """
    def __init__(self, file_path: str, jq_schema: str, json_lines: bool = False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSON file not found at {file_path}")
        self.file_path = file_path
        self.jq_schema = jq_schema
        self.json_lines = json_lines

    def load(self) -> List[Document]:
        try:
            loader = JSONLoader(
                file_path=self.file_path,
                jq_schema=self.jq_schema,
                json_lines=self.json_lines,
                text_content=False  # Get metadata as well
            )
            documents = loader.load()
            for doc in documents:
                doc.metadata.setdefault("source_file_basename", os.path.basename(self.file_path))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load JSON from {self.file_path} with schema '{self.jq_schema}': {e}")


class DataFrameSource(DataSource):
    """Concrete data source for pandas DataFrames."""
    def __init__(self, dataframe: pd.DataFrame, page_content_column: str, metadata_columns: Optional[List[str]] = None):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if page_content_column not in dataframe.columns:
            raise ValueError(f"Page content column '{page_content_column}' not found in DataFrame.")
        
        self.dataframe = dataframe
        self.page_content_column = page_content_column
        self.metadata_columns = metadata_columns or []

        for col in self.metadata_columns:
            if col not in dataframe.columns:
                raise ValueError(f"Metadata column '{col}' not found in DataFrame.")

    def load(self) -> List[Document]:
        try:
            documents = []
            for _, row in self.dataframe.iterrows():
                page_content = str(row[self.page_content_column])
                metadata = {"source_type": "pandas_dataframe"}
                for col in self.metadata_columns:
                    metadata[col] = row[col]
                documents.append(Document(page_content=page_content, metadata=metadata))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load data from DataFrame: {e}")


class PickleSource(DataSource):
    """
    Concrete data source for Pickle files.
    Assumes the pickle file contains: a single string, a list of strings,
    a single Langchain Document, or a list of Langchain Document objects.
    WARNING: Only load pickle files from trusted sources due to security risks.
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Pickle file not found at {file_path}")
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            
            documents = []
            source_name = os.path.basename(self.file_path)
            base_metadata = {"source": source_name, "source_full_path": self.file_path}

            if isinstance(data, list):
                if all(isinstance(item, Document) for item in data):
                    for doc in data:
                        doc.metadata.update(base_metadata) # Ensure our metadata is there
                        documents.append(doc)
                elif all(isinstance(item, str) for item in data):
                    for text_content in data:
                        documents.append(Document(page_content=text_content, metadata=dict(base_metadata)))
                else:
                    raise ValueError("Pickled list must contain either all Document objects or all strings.")
            elif isinstance(data, Document):
                data.metadata.update(base_metadata)
                documents.append(data)
            elif isinstance(data, str):
                documents.append(Document(page_content=data, metadata=dict(base_metadata)))
            else:
                raise ValueError("Unsupported data type in pickle file. Expected Document, str, List[Document], or List[str].")
            return documents
        except pickle.UnpicklingError as e:
            raise DataSourceError(f"Failed to unpickle file {self.file_path}: {e}. Not a valid pickle file.")
        except ValueError as e: # Catch custom ValueErrors
            raise DataSourceError(f"Error processing pickled data from {self.file_path}: {e}")
        except Exception as e:
            raise DataSourceError(f"Failed to load Pickle from {self.file_path}: {e}")


class ParquetSource(DataSource):
    """
    Concrete data source for Parquet files.
    Reads the Parquet file into a pandas DataFrame and then converts specified
    column(s) to Langchain Documents.
    """
    def __init__(self, file_path: str, page_content_column: str, metadata_columns: Optional[List[str]] = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found at {file_path}")
        self.file_path = file_path
        self.page_content_column = page_content_column
        self.metadata_columns = metadata_columns or []

    def load(self) -> List[Document]:
        try:
            df = pd.read_parquet(self.file_path)
            
            if self.page_content_column not in df.columns:
                raise ValueError(f"Page content column '{self.page_content_column}' not found in Parquet file.")
            for col in self.metadata_columns:
                if col not in df.columns:
                    raise ValueError(f"Metadata column '{col}' not found in Parquet file.")

            documents = []
            source_name = os.path.basename(self.file_path)
            for _, row in df.iterrows():
                page_content = str(row[self.page_content_column])
                metadata = {"source": source_name, "source_full_path": self.file_path}
                for col in self.metadata_columns:
                    metadata[col] = row[col]
                documents.append(Document(page_content=page_content, metadata=metadata))
            return documents
        except ValueError as e:
            raise DataSourceError(f"Configuration error for Parquet file {self.file_path}: {e}")
        except Exception as e:
            raise DataSourceError(f"Failed to load Parquet from {self.file_path}: {e}")


class TextFileSource(DataSource):
    """Concrete data source for generic text-based files (e.g., .txt, .md, .html) using UnstructuredFileLoader."""
    def __init__(self, file_path: str, unstructured_kwargs: Optional[Dict[str, Any]] = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        self.file_path = file_path
        self.unstructured_kwargs = unstructured_kwargs or {}

    def load(self) -> List[Document]:
        try:
            loader = UnstructuredFileLoader(self.file_path, **self.unstructured_kwargs)
            documents = loader.load()
            for doc in documents:
                doc.metadata.setdefault("source_file_basename", os.path.basename(self.file_path))
            return documents
        except Exception as e:
            raise DataSourceError(f"Failed to load file {self.file_path} using UnstructuredFileLoader: {e}")