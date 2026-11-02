import { useState } from 'react'
import IndexPanel from './components/IndexPanel'
import ChatPanel from './components/ChatPanel'
import SourcesPanel from './components/SourcesPanel'
import './App.css'

/**
 * App layout:
 *
 * ┌──────────────┬──────────────────────┬─────────────────┐
 * │  IndexPanel  │      ChatPanel       │  SourcesPanel   │
 * │  (sidebar)   │  (flex-1, center)    │  (right panel)  │
 * └──────────────┴──────────────────────┴─────────────────┘
 */
export default function App() {
  const [isIndexed, setIsIndexed] = useState(false)
  const [sources, setSources] = useState([])

  function handleIndexed(data) {
    setIsIndexed(true)
    setSources([]) // clear previous sources on re-index
    // If the index result carries chunk info, mark as indexed
    if (data && (data.total_chunks > 0 || data.is_indexed)) {
      setIsIndexed(true)
    }
  }

  return (
    <div className="flex h-screen bg-gray-950 text-white overflow-hidden">
      {/* Left sidebar — IndexPanel */}
      <IndexPanel onIndexed={handleIndexed} />

      {/* Center — Chat */}
      <main className="flex-1 flex flex-col min-w-0 border-r border-gray-700">
        <header className="shrink-0 px-4 py-3 border-b border-gray-700 bg-gray-900/70 backdrop-blur flex items-center gap-2">
          <h2 className="text-sm font-semibold text-gray-200">Chat</h2>
          {isIndexed && (
            <span className="text-[10px] rounded-full px-2 py-0.5 bg-green-900/60 border border-green-700 text-green-400">
              indexed
            </span>
          )}
        </header>
        <div className="flex-1 overflow-hidden">
          <ChatPanel onSources={setSources} isIndexed={isIndexed} />
        </div>
      </main>

      {/* Right sidebar — SourcesPanel */}
      <aside className="w-80 min-w-64 flex flex-col bg-gray-900 overflow-hidden shrink-0">
        <header className="shrink-0 px-4 py-3 border-b border-gray-700 bg-gray-900/70">
          <h2 className="text-sm font-semibold text-gray-200">Sources</h2>
        </header>
        <div className="flex-1 overflow-y-auto">
          <SourcesPanel sources={sources} />
        </div>
      </aside>
    </div>
  )
}
