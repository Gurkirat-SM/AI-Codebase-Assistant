import { useState, useEffect } from 'react'
import { getIndexStatus, indexRepository } from '../api'

/**
 * IndexPanel — left sidebar for pointing at a repository and indexing it.
 */
export default function IndexPanel({ onIndexed }) {
  const [repoPath, setRepoPath] = useState('')
  const [status, setStatus] = useState(null)   // { is_indexed, total_chunks, total_files, ... }
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(null)  // index result

  // Load current index status on mount
  useEffect(() => {
    getIndexStatus()
      .then(r => setStatus(r.data))
      .catch(() => setStatus(null))
  }, [])

  async function handleIndex(e) {
    e.preventDefault()
    if (!repoPath.trim()) return
    setError('')
    setSuccess(null)
    setLoading(true)
    try {
      const res = await indexRepository(repoPath.trim())
      setSuccess(res.data)
      setStatus({ is_indexed: true, ...res.data })
      if (onIndexed) onIndexed(res.data)
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'Indexing failed'
      setError(String(msg))
    } finally {
      setLoading(false)
    }
  }

  return (
    <aside className="flex flex-col gap-4 p-4 w-72 min-w-64 bg-gray-900 border-r border-gray-700 overflow-y-auto shrink-0">
      {/* Header */}
      <div>
        <h1 className="text-lg font-semibold text-white flex items-center gap-2">
          <span>🔍</span>
          <span>AI Codebase Assistant</span>
        </h1>
        <p className="text-xs text-gray-400 mt-1">
          Index a repository, then ask questions about it.
        </p>
      </div>

      <hr className="border-gray-700" />

      {/* Index form */}
      <form onSubmit={handleIndex} className="flex flex-col gap-3">
        <label className="text-sm font-medium text-gray-300">
          Repository path
        </label>
        <input
          type="text"
          value={repoPath}
          onChange={e => setRepoPath(e.target.value)}
          placeholder="C:\path\to\your\repo"
          disabled={loading}
          className="px-3 py-2 rounded-md bg-gray-800 border border-gray-600 text-sm text-white placeholder-gray-500
                     focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                     disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={loading || !repoPath.trim()}
          className="flex items-center justify-center gap-2 px-4 py-2 rounded-md
                     bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700
                     text-white text-sm font-medium
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
        >
          {loading ? (
            <>
              <Spinner />
              Indexing…
            </>
          ) : (
            'Index Repository'
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="rounded-md bg-red-900/40 border border-red-700 p-3 text-xs text-red-300">
          ⚠️ {error}
        </div>
      )}

      {/* Success */}
      {success && (
        <div className="rounded-md bg-green-900/40 border border-green-700 p-3 text-xs text-green-300 space-y-1">
          <p className="font-semibold">✅ Indexed successfully!</p>
          {success.total_files != null && (
            <p>Files: <span className="text-green-200 font-mono">{success.total_files}</span></p>
          )}
          {success.total_chunks != null && (
            <p>Chunks: <span className="text-green-200 font-mono">{success.total_chunks}</span></p>
          )}
        </div>
      )}

      {/* Current status */}
      {status && !success && (
        <div className="rounded-md bg-gray-800 border border-gray-700 p-3 text-xs text-gray-400 space-y-1">
          <p className="font-semibold text-gray-300">Current index</p>
          <p>
            Status:{' '}
            <span className={status.is_indexed ? 'text-green-400' : 'text-yellow-400'}>
              {status.is_indexed ? 'Indexed' : 'Not indexed'}
            </span>
          </p>
          {status.total_chunks != null && (
            <p>Chunks: <span className="font-mono text-gray-300">{status.total_chunks}</span></p>
          )}
          {status.total_files != null && (
            <p>Files: <span className="font-mono text-gray-300">{status.total_files}</span></p>
          )}
          {status.repo_path && (
            <p className="break-all">
              Path: <span className="font-mono text-gray-300">{status.repo_path}</span>
            </p>
          )}
        </div>
      )}

      {/* Tips */}
      <div className="mt-auto rounded-md bg-gray-800/60 border border-gray-700 p-3 text-xs text-gray-500 space-y-1">
        <p className="font-medium text-gray-400">Tips</p>
        <p>• Use an absolute path to the repo root.</p>
        <p>• Re-index anytime to pick up new changes.</p>
        <p>• Indexing may take a minute for large repos.</p>
      </div>
    </aside>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
    </svg>
  )
}

