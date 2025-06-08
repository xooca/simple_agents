# agentic_orchestrator/data_ingestion/loader.py
import os
from typing import Any, Optional, Dict
import pandas as pd

from .base import DataSource
from .sources import (
    PDFSource,
    CSVSource,
    WordSource,
    JSONSource,
    DataFrameSource,
    PickleSource,
    ParquetSource,
    TextFileSource,
)
from ..exceptions import UnsupportedDataSourceError, DataSourceError


class DataLoaderFactory:
    """Factory class to create an appropriate DataSource instance based on input."""

    @staticmethod
    def get_loader(
        source_type: Optional[str] = None,
        file_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> DataSource:
        """
        Gets a data loader based on the source type or file extension.

        Args:
            source_type (Optional[str]): Explicit type of the source (e.g., "pdf", "csv", "dataframe", "text").
                                         If None, type is inferred from file_path extension.
            file_path (Optional[str]): Path to the data file. Required if not a DataFrame source.
            dataframe (Optional[pd.DataFrame]): Pandas DataFrame to load. Required for "dataframe" source_type.
            **kwargs: Additional arguments specific to the loader (e.g., jq_schema for JSON, 
                      page_content_column for DataFrame/Parquet, unstructured_kwargs for TextFileSource).

        Returns:
            DataSource: An instance of a DataSource.

        Raises:
            UnsupportedDataSourceError: If the source type is not supported or cannot be inferred.
            ValueError: If required arguments for a specific loader are missing.
            FileNotFoundError: If file_path is provided but the file does not exist.
        """

        if source_type == "dataframe":
            if dataframe is None:
                raise ValueError("A pandas DataFrame must be provided for 'dataframe' source_type.")
            page_content_column = kwargs.get("page_content_column")
            if not page_content_column:
                raise ValueError("'page_content_column' is required for DataFrameSource.")
            return DataFrameSource(
                dataframe=dataframe,
                page_content_column=page_content_column,
                metadata_columns=kwargs.get("metadata_columns")
            )

        if not file_path:
            raise ValueError("file_path must be provided if source_type is not 'dataframe'.")
        
        # File existence is checked by individual Source constructors,
        # but an early check here can be useful.
        # if not os.path.exists(file_path):
        #     raise FileNotFoundError(f"File not found at {file_path}")

        inferred_type = ""
        if file_path:
            _, extension = os.path.splitext(file_path.lower())
            inferred_type = extension[1:]  # remove dot

        load_type = source_type if source_type else inferred_type

        try:
            if load_type == "pdf":
                return PDFSource(file_path)
            elif load_type == "csv":
                return CSVSource(
                    file_path,
                    source_column=kwargs.get("source_column"),
                    csv_args=kwargs.get("csv_args")
                )
            elif load_type in ("doc", "docx"):
                return WordSource(file_path)
            elif load_type == "json":
                jq_schema = kwargs.get("jq_schema")
                if not jq_schema:
                    raise ValueError("'jq_schema' is required for JSONSource.")
                return JSONSource(
                    file_path,
                    jq_schema=jq_schema,
                    json_lines=kwargs.get("json_lines", False)
                )
            elif load_type in ("pkl", "pickle"):
                return PickleSource(file_path)
            elif load_type == "parquet":
                page_content_column = kwargs.get("page_content_column")
                if not page_content_column:
                    raise ValueError("'page_content_column' is required for ParquetSource.")
                return ParquetSource(
                    file_path,
                    page_content_column=page_content_column,
                    metadata_columns=kwargs.get("metadata_columns")
                )
            elif load_type in ("txt", "md", "html", "xml", "text") or (not load_type and os.path.isfile(file_path)):
                # "text" as an explicit type, or fallback for recognized extensions or any file
                return TextFileSource(file_path, unstructured_kwargs=kwargs.get("unstructured_kwargs"))
            else:
                raise UnsupportedDataSourceError(f"Unsupported or unrecognized data source type: '{load_type}' for file: {file_path}")
        except FileNotFoundError as e: # Catch FileNotFoundError from Source constructors
            raise e
        except Exception as e: # Catch other errors from Source constructors (ValueError, etc.)
            raise DataSourceError(f"Error initializing loader for {file_path} (type: {load_type}): {e}")