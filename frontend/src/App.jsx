import { useEffect, useMemo, useRef, useState } from 'react';
import { sendChatMessage } from './services/chatApi';

const suggestions = [
  'Is it safe to go running in Ahmedabad right now?',
  'Can I cycle to work today?',
  'Is it safe to fly a kite this afternoon?',
  'Can I have a picnic this evening?',
];

function getSessionId() {
  const key = 'weather-advisory-session-id';
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(key, id);
  }
  return id;
}

function weatherLabel(requestedTime) {
  const value = (requestedTime || '').toLowerCase();
  if (value.includes('afternoon')) return 'Weather for this afternoon';
  if (
    value.includes('evening') ||
    value.includes('later') ||
    value.includes('night')
  ) {
    return 'Weather for this evening';
  }
  if (value.includes('tomorrow')) return 'Weather for tomorrow';
  if (value.includes('now') || value.includes('current')) {
    return 'Current weather';
  }
  return 'Weather used for this advisory';
}

function formatTimestamp(value) {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(parsed);
}

function WeatherSummary({ weather, requestedTime, hasPolicy }) {
  if (!weather) return null;

  const values = [
    ['Temperature', `${weather.temperature_c}°C`],
    ['Wind', `${weather.wind_speed_kmh} km/h`],
    ['Rain', `${weather.precipitation_mm} mm`],
    ['Rain chance', `${weather.precipitation_probability_pct}%`],
    ['UV', weather.uv_index],
    [
      'Condition',
      weather.weather_condition
        ? weather.weather_condition[0].toUpperCase() + weather.weather_condition.slice(1)
        : '—',
    ],
    ['Timestamp', formatTimestamp(weather.timestamp)],
  ];

  return (
    <section className="weather-meta" aria-label={weatherLabel(requestedTime)}>
      <div className="section-label">
        {hasPolicy ? weatherLabel(requestedTime) : 'Weather checked'}
      </div>
      <div className="weather-grid">
        {values.map(([label, value]) => (
          <div className="weather-value" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function PolicySummary({ data }) {
  if (data.error_type || !data.sop_id) return null;
  return (
    <div className="policy-line" aria-label="Policy used">
      <span>{data.sop_id}</span>
      {data.severity && (
        <span className={`severity ${data.severity.toLowerCase()}`}>
          {data.severity}
        </span>
      )}
    </div>
  );
}

function Message({ message }) {
  if (message.role === 'user') {
    return (
      <div className="message user">
        <p>{message.content}</p>
      </div>
    );
  }

  return (
    <article className={`message assistant ${message.error ? 'error' : ''}`}>
      <div className="assistant-label">Weather Advisory</div>
      <p>{message.content}</p>
      {message.data && (
        <>
          <WeatherSummary
            weather={message.data.weather}
            requestedTime={message.data.requested_time}
            hasPolicy={Boolean(message.data.sop_id)}
          />
          <PolicySummary data={message.data} />
        </>
      )}
    </article>
  );
}

function EmptyState({ onSuggestion }) {
  return (
    <div className="empty-state">
      <h2>Ask about outdoor conditions</h2>
      <p>
        Get guidance based on current weather and the project's written safety policies.
      </p>
      <div className="suggestions">
        {suggestions.map((prompt) => (
          <button
            type="button"
            className="suggestion"
            key={prompt}
            onClick={() => onSuggestion(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

function Composer({ value, onChange, onSubmit, loading }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = 'auto';
      ref.current.style.height = `${Math.min(ref.current.scrollHeight, 144)}px`;
    }
  }, [value]);

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSubmit(event);
    }
  }

  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        ref={ref}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type your weather or activity question..."
        disabled={loading}
        rows="1"
        aria-label="Weather question"
      />
      <button
        className="send"
        aria-label="Send question"
        disabled={loading || !value.trim()}
      >
        <span aria-hidden="true">↑</span>
      </button>
      <div className="composer-hint">Enter to send · Shift + Enter for a new line</div>
    </form>
  );
}

export default function App() {
  const sessionId = useMemo(getSessionId, []);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const conversationRef = useRef(null);

  useEffect(() => {
    const node = conversationRef.current;
    if (node) {
      node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, loading]);

  async function submit(event, suggestedMessage) {
    event?.preventDefault();
    const message = (suggestedMessage ?? input).trim();
    if (!message || loading) return;
    setInput('');
    setMessages((current) => [...current, { role: 'user', content: message }]);
    setLoading(true);
    try {
      const data = await sendChatMessage(sessionId, message);
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: data.answer || data.error || 'No response received.',
          data,
          error: Boolean(data.error_type),
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: error.message, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-title">Weather Advisory</div>
      </header>
      <section className="conversation" ref={conversationRef} aria-live="polite">
        {messages.length === 0 ? (
          <EmptyState onSuggestion={(prompt) => submit(null, prompt)} />
        ) : (
          <div className="reading-column">
            {messages.map((message, index) => (
              <Message
                message={message}
                key={`${message.role}-${index}`}
              />
            ))}
            {loading && (
              <div className="message assistant loading-message">
                <div className="assistant-label">Weather Advisory</div>
                <p>
                  <span className="thinking-dots">
                    <i /> <i /> <i />
                  </span>
                  <span className="sr-only">Checking live weather</span>
                </p>
              </div>
            )}
          </div>
        )}
      </section>
      <Composer
        value={input}
        onChange={setInput}
        onSubmit={submit}
        loading={loading}
      />
    </main>
  );
}
