/**
 * SourcesPanel — shows the retrieved code chunks for the latest assistant answer.
 * sources: Array<{ file_path, start_line, end_line, snippet, score, language }>
 */
export default function SourcesPanel({ sources }) {
  if (!sources || sources.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-gray-500 p-6 text-center">
        <p>Source chunks will appear here after you ask a question.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 p-4 overflow-y-auto h-full">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-400">
        Sources ({sources.length})
      </h2>
      {sources.map((src, i) => (
        <SourceCard key={i} source={src} index={i + 1} />
      ))}
    </div>
  )
}

function SourceCard({ source, index }) {
  const { file_path, start_line, end_line, snippet, score, language } = source
  const scorePct = Math.round((score ?? 0) * 100)

  // Colour-code by similarity score
  const scoreColor =
    scorePct >= 70
      ? 'text-green-400 bg-green-900/30 border-green-700/50'
      : scorePct >= 40
      ? 'text-yellow-400 bg-yellow-900/30 border-yellow-700/50'
      : 'text-orange-400 bg-orange-900/30 border-orange-700/50'

  const barColor =
    scorePct >= 70 ? 'bg-green-500' : scorePct >= 40 ? 'bg-yellow-500' : 'bg-orange-500'

  const shortPath = file_path ? file_path.replace(/\\/g, '/') : ''

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/60 overflow-hidden text-xs">
      {/* Card header */}
      <div className="flex items-start justify-between gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-gray-500 shrink-0">#{index}</span>
          <span
            className="text-indigo-300 font-mono truncate"
            title={shortPath}
          >
            {shortPath}
          </span>
        </div>
        {/* Line range badge */}
        {start_line != null && (
          <span className="shrink-0 rounded px-1.5 py-0.5 bg-gray-700 text-gray-300 font-mono whitespace-nowrap">
            L{start_line}–{end_line ?? start_line}
          </span>
        )}
      </div>

      {/* Score bar */}
      <div className="px-3 pt-2 pb-1 flex items-center gap-2">
        <div className="flex-1 h-1.5 rounded-full bg-gray-700 overflow-hidden">
          <div
            className={`h-full rounded-full ${barColor} transition-all`}
            style={{ width: `${scorePct}%` }}
          />
        </div>
        <span className={`rounded px-1.5 py-0.5 border text-[10px] font-semibold ${scoreColor}`}>
          {scorePct}%
        </span>
        {language && (
          <span className="rounded px-1.5 py-0.5 bg-gray-700 text-gray-400 text-[10px]">
            {language}
          </span>
        )}
      </div>

      {/* Snippet */}
      {snippet && (
        <details className="group">
          <summary className="cursor-pointer select-none px-3 py-1.5 text-gray-400 hover:text-gray-200 transition-colors flex items-center gap-1">
            <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
            <span>Show snippet</span>
          </summary>
          <pre className="m-0 px-3 py-2 bg-gray-900 border-t border-gray-700 text-gray-300 text-[11px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-words font-mono">
            {snippet}
          </pre>
        </details>
      )}
    </div>
  )
}

