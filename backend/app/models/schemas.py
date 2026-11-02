from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = "ok"
    version: str
    project_name: str


class FileInfo(BaseModel):
    """Metadata for an indexed or scanned repository file."""
    relative_path: str
    absolute_path: str
    size_bytes: int
    line_count: int
    extension: str
    language: str


class ScanRequest(BaseModel):
    """Request payload to scan a repository."""
    repo_path: str = Field(..., description="Absolute or relative path to the repository directory")
    additional_exclusions: List[str] = Field(
        default_factory=list,
        description="Optional additional patterns or directories to exclude"
    )
    max_file_size_bytes: Optional[int] = Field(
        default=None,
        description="Override maximum allowable file size in bytes"
    )


class ScanSummary(BaseModel):
    """Summary of a repository scan."""
    scanned_path: str
    total_files: int
    total_lines: int
    total_size_bytes: int
    files_by_language: Dict[str, int]
    files: List[FileInfo]


class CodeChunk(BaseModel):
    """A line-preserving segment of code with line number tracking."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. path:start_line-end_line")
    file_path: str = Field(..., description="Relative file path")
    start_line: int = Field(..., ge=1, description="1-based starting line number")
    end_line: int = Field(..., ge=1, description="1-based ending line number")
    content: str = Field(..., description="Exact content of the chunk preserving line breaks and formatting")
    language: str = Field(default="Unknown", description="Programming or markup language")
    line_count: int = Field(..., ge=1, description="Number of lines in the chunk")
    chunk_index: int = Field(..., ge=0, description="0-based index of this chunk within the file")


class ChunkRequest(BaseModel):
    """Request payload to chunk raw text or a single file."""
    file_path: Optional[str] = Field(default="snippet.txt", description="Display path for the snippet")
    content: str = Field(..., description="Text or source code to chunk")
    language: Optional[str] = Field(default=None, description="Programming language if known")
    chunk_size_lines: Optional[int] = Field(default=None, ge=5, le=500, description="Chunk line size limit")
    chunk_overlap_lines: Optional[int] = Field(default=None, ge=0, le=200, description="Lines of overlap between adjacent chunks")


class ChunkResponse(BaseModel):
    """Response containing code chunks from a file or text."""
    file_path: str
    language: str
    total_lines: int
    total_chunks: int
    chunks: List[CodeChunk]


class ScanAndChunkRequest(BaseModel):
    """Request to scan a repository and generate chunks for all eligible files."""
    repo_path: str = Field(..., description="Path to the repository directory")
    additional_exclusions: List[str] = Field(default_factory=list, description="Extra exclusion patterns")
    max_file_size_bytes: Optional[int] = Field(default=None, description="File size threshold")
    chunk_size_lines: Optional[int] = Field(default=None, ge=5, le=500, description="Chunk line size")
    chunk_overlap_lines: Optional[int] = Field(default=None, ge=0, le=200, description="Overlap lines")


class ScanAndChunkResponse(BaseModel):
    """Response containing repository scan summary and all parsed chunks."""
    scan_summary: ScanSummary
    total_chunks: int
    chunks: List[CodeChunk]


class QueryResult(BaseModel):
    """A retrieved code chunk paired with its semantic similarity score."""
    chunk: CodeChunk
    score: float = Field(..., description="Cosine similarity score (between 0.0 and 1.0)")


class QuerySource(BaseModel):
    """Retrieved source metadata, snippet, and similarity score."""
    file_path: str = Field(..., description="Relative path to file")
    start_line: int = Field(..., ge=1, description="1-based starting line number")
    end_line: int = Field(..., ge=1, description="1-based ending line number")
    snippet: str = Field(..., description="Code content snippet")
    score: float = Field(..., description="Cosine similarity score")
    language: str = Field(default="Unknown", description="Language of snippet")


class QueryRequest(BaseModel):
    """Request payload for semantic code retrieval."""
    query: str = Field(..., min_length=1, description="Natural language or code query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of chunks to retrieve")
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score threshold",
    )
    file_filter: Optional[str] = Field(
        default=None,
        description="Optional substring filter for relative file paths",
    )


class QueryResponse(BaseModel):
    """Response payload containing generated answer and retrieved sources."""
    query: str
    answer: str = Field(default="", description="Generated natural-language answer based on retrieved context")
    sources: List[QuerySource] = Field(default_factory=list, description="Retrieved code sources")
    total_results: int = Field(default=0, description="Total sources retrieved")
    results: List[QueryResult] = Field(default_factory=list, description="Raw chunk results for backward compatibility")


class IndexRequest(BaseModel):
    """Request payload to scan, chunk, embed, and index a repository."""
    repo_path: str = Field(..., description="Path to repository to index")
    persist: bool = Field(default=True, description="Whether to persist the FAISS index to disk")
    additional_exclusions: List[str] = Field(default_factory=list, description="Extra exclusion patterns")
    chunk_size_lines: Optional[int] = Field(default=None, ge=5, le=500, description="Chunk line size")
    chunk_overlap_lines: Optional[int] = Field(default=None, ge=0, le=200, description="Overlap lines")


class IndexResponse(BaseModel):
    """Response payload summarizing the indexing operation."""
    status: str = "success"
    message: str
    repo_path: str
    total_files: int
    total_chunks: int
    embedding_dimension: int
    model_name: str
    persisted: bool
    index_storage_dir: Optional[str] = None


class IndexStatusResponse(BaseModel):
    """Status of the in-memory or persisted FAISS index."""
    is_indexed: bool
    repo_path: Optional[str] = None
    total_chunks: int
    embedding_dimension: Optional[int] = None
    model_name: str
    persisted_dir: Optional[str] = None


