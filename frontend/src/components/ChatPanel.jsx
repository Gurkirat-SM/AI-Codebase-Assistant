import { useState, useRef, useEffect } from 'react'
import { queryRepository } from '../api'
import MessageBubble from './MessageBubble'

/**
 * ChatPanel — main chat interface.
 * onSources: callback fired with the sources array after each successful query.
 * isIndexed: boolean — whether the repo has been indexed.
 */
export default function ChatPanel({ onSources, isIndexed }) {
  const [messages, setMessages] = useState([])   // [{ role, content }]
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [topK] = useState(5)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to newest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setError('')

    // Append user message immediately
    setMessages(prev => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const res = await queryRepository(text, topK)
      const data = res.data

      const answer = data.answer || '*(No answer returned)*'
      setMessages(prev => [...prev, { role: 'assistant', content: answer }])

      if (onSources && data.sources) {
        onSources(data.sources)
      }
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'Query failed'
      setError(String(msg))
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `⚠️ Error: ${String(msg)}` },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <span className="text-4xl">💬</span>
            <p className="text-gray-400 text-sm max-w-sm">
              {isIndexed
                ? 'Your repository is indexed. Ask a question about the code!'
                : 'Index a repository first using the panel on the left, then ask a question.'}
            </p>
            {isIndexed && (
              <div className="flex flex-wrap justify-center gap-2 mt-2">
                {[
                  'How does the main app work?',
                  'What does the scanner exclude?',
                  'How are embeddings generated?',
                  'Explain the query endpoint',
                ].map(hint => (
                  <button
                    key={hint}
                    onClick={() => { setInput(hint); inputRef.current?.focus() }}
                    className="px-3 py-1.5 rounded-full bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 transition-colors"
                  >
                    {hint}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} />
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 border border-gray-700 rounded-2xl rounded-bl-sm px-4 py-3">
              <TypingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-gray-700 px-4 py-3">
        {error && (
          <p className="text-xs text-red-400 mb-2">⚠️ {error}</p>
        )}
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isIndexed ? 'Ask about your code…' : 'Index a repo first…'}
            disabled={loading || !isIndexed}
            rows={1}
            className="flex-1 resize-none px-3 py-2 rounded-xl bg-gray-800 border border-gray-600 text-sm text-white placeholder-gray-500
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                       disabled:opacity-50 disabled:cursor-not-allowed
                       max-h-40 overflow-y-auto leading-relaxed"
            style={{ height: 'auto', minHeight: '40px' }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = e.target.scrollHeight + 'px'
            }}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !isIndexed || !input.trim()}
            className="shrink-0 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700
                       text-white text-sm font-medium
                       disabled:opacity-50 disabled:cursor-not-allowed
                       transition-colors h-10 flex items-center"
          >
            {loading ? '…' : 'Send'}
          </button>
        </div>
        <p className="text-[10px] text-gray-600 mt-1.5 text-right">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}

function TypingDots() {
  return (
    <span className="flex gap-1 items-center h-4">
      {[0, 1, 2].map(i => (
        <span
          key={i}
          className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </span>
  )
}

