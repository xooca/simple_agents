# agentic_orchestrator/data_ingestion/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class DataSource(ABC):
    """Abstract base class for a data source."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Loads data from the source and returns a list of LangChain Documents."""
        pass

