import React, { useEffect, useRef, useState } from 'react';
import { Bot, Sparkles, Send, Plus, ExternalLink, LoaderCircle, Wifi, WifiOff } from 'lucide-react';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const suggestions = [
  'What is the current news of Andhra Pradesh?',
  'Explain how RAG works with a simple example',
  'Compare FastAPI and Node.js for an AI backend',
  'Find current fresher AI Engineer job openings in India'
];

function renderInline(text) {
  const parts = [];
  const pattern = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)|\*\*([^*]+)\*\*/g;
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));
    if (match[1] && match[2]) {
      parts.push(<a href={match[2]} target="_blank" rel="noreferrer" key={`link-${match.index}`}>{match[1]}</a>);
    } else if (match[3]) {
      parts.push(<strong key={`bold-${match.index}`}>{match[3]}</strong>);
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => setConnected(r.ok))
      .catch(() => setConnected(false));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function sendMessage(value = input) {
    const text = value.trim();
    if (!text || loading) return;
    const userMessage = { role: 'user', content: text };
    setMessages(old => [...old, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The request failed.');
      setMessages(old => [
        ...old,
        {
          role: 'assistant',
          content: data.answer || 'No answer returned.',
          status: data.status,
          note: data.completion_note,
          agents: data.agents || [],
          sources: data.sources || [],
        },
      ]);
      setConnected(true);
    } catch (error) {
      setMessages(old => [
        ...old,
        {
          role: 'assistant',
          content: `I couldn't connect to AgentFlow AI. ${error.message}\n\nMake sure the FastAPI backend is running at ${API}.`,
          error: true,
        },
      ]);
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([]);
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="app-shell">
      <header className="chat-header">
        <div className="brand">
          <div className="brand-mark"><Sparkles size={19}/></div>
          <div>
            <strong>AgentFlow AI</strong>
            <span>Intelligent multi-agent assistant</span>
          </div>
        </div>
        <div className="header-actions">
          <span className={`connection-pill ${connected ? 'online' : 'offline'}`}>
            {connected ? <Wifi size={14}/> : <WifiOff size={14}/>} {connected ? 'Connected' : 'Offline'}
          </span>
          <button className="new-chat" onClick={newChat}>
            <Plus size={16}/> New chat
          </button>
        </div>
      </header>

      <main className="chat-main">
        <section className={`conversation ${messages.length ? 'has-messages' : ''}`}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">
                <Bot size={30}/><span><Sparkles size={13}/></span>
              </div>
              <p className="eyebrow">AGENTFLOW AI</p>
              <h1>How can I help you today?</h1>
              <p className="welcome-sub">
                Ask a question, explore a topic, research current events or work through a task. AgentFlow will bring you a relevant answer.
              </p>
              <div className="suggestions">
                {suggestions.map((item, index) => (
                  <button key={index} onClick={() => sendMessage(item)}>
                    {item}<span>↗</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => (
                <article className={`message ${message.role}`} key={index}>
                  <div className={`avatar ${message.role}`}>
                    {message.role === 'user' ? 'You' : <Bot size={18}/>}
                  </div>
                  <div className="message-body">
                    <div className="message-author">{message.role === 'user' ? 'You' : 'AgentFlow AI'}</div>
                    <div className="answer-text">
                      {message.content.split('\n').map((line, lineIndex) =>
                        line.startsWith('### ') ? (
                          <h4 key={lineIndex}>{renderInline(line.slice(4))}</h4>
                        ) : line.startsWith('## ') ? (
                          <h3 key={lineIndex}>{renderInline(line.slice(3))}</h3>
                        ) : line.startsWith('# ') ? (
                          <h2 key={lineIndex}>{renderInline(line.slice(2))}</h2>
                        ) : line.startsWith('- ') || line.startsWith('* ') ? (
                          <p className="bullet" key={lineIndex}>• {renderInline(line.slice(2))}</p>
                        ) : line ? (
                          <p key={lineIndex}>{renderInline(line)}</p>
                        ) : (
                          <div className="spacer-line" key={lineIndex}/>
                        )
                      )}
                    </div>
                    {message.sources?.length > 0 && (
                      <div className="source-list">
                        <strong>Sources</strong>
                        {message.sources.map((source, sourceIndex) => (
                          <a href={source.url} target="_blank" rel="noreferrer" key={sourceIndex}>
                            {source.title || source.url}
                            <ExternalLink size={12}/>
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}

          {loading && (
            <div className="loading-row">
              <div className="avatar assistant"><Bot size={18}/></div>
              <div>
                <div className="loading-title"><LoaderCircle size={15}/> AgentFlow is thinking</div>
                <div className="loading-sub">Preparing a relevant answer…</div>
              </div>
            </div>
          )}
          <div ref={endRef}/>
        </section>

        <footer className="composer-wrap">
          <div className="composer">
            <textarea
              value={input}
              onChange={event => setInput(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask AgentFlow AI anything…"
              rows={1}
              disabled={loading}
            />
            <div className="composer-bottom">
              <span>Enter to send · Shift + Enter for a new line</span>
              <button
                className="send-button"
                onClick={() => sendMessage()}
                disabled={!input.trim() || loading}
                aria-label="Send message"
              >
                <Send size={17}/>
              </button>
            </div>
          </div>
          <p className="disclaimer">
            AgentFlow AI may make mistakes. Current and important information is checked against retrieved sources.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
