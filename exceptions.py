# agentic_orchestrator/exceptions.py
class AgenticOrchestratorError(Exception):
    """Base exception for the package."""
    pass

class DataSourceError(AgenticOrchestratorError):
    """Error during data ingestion."""
    pass

class UnsupportedDataSourceError(DataSourceError):
    """Error when a data source type is not supported."""
    pass

class DataStoreError(AgenticOrchestratorError):
    """Error during data storage or retrieval."""
    pass

class AgentExecutionError(AgenticOrchestratorError):
    """Error during agent execution."""
    pass