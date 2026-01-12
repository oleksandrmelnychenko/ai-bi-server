import { useEffect, useMemo, useRef, useState } from "react";


type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  sql?: string;
  columns?: string[];
  rows?: Array<Array<unknown>>;
  warnings?: string[];
};

type ChatResponse = {
  answer: string;
  sql?: string | null;
  columns?: string[];
  rows?: Array<Array<unknown>>;
  warnings?: string[];
};

const makeId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Вітаю! Поставте питання українською, і я сформую SQL-запит до вашої бази даних та поверну відповідь."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const compactHistory = useMemo(
    () =>
      messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .slice(-6)
        .map((message) => ({ role: message.role, content: message.content })),
    [messages]
  );

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed || loading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: trimmed
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history: compactHistory })
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Request failed");
      }

      const data = (await response.json()) as ChatResponse;
      const assistantMessage: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: data.answer,
        sql: data.sql || undefined,
        columns: data.columns || undefined,
        rows: data.rows || undefined,
        warnings: data.warnings || undefined
      };
      setMessages([...nextMessages, assistantMessage]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "assistant",
          content: "Сталася помилка під час запиту. Перевірте сервер і спробуйте ще раз."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Concord Insight</p>
          <h1>Живий діалог з даними</h1>
          <p className="subtitle">
            Напишіть запит українською. Система сформує складні JOIN-и та поверне відповідь.
          </p>
        </div>
        <div className="status">
          <span className="pulse" />
          <span>LLM + SQL Server</span>
        </div>
      </header>

      <section className="chat">
        <div className="messages">
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`message ${message.role}`}
              style={{ animationDelay: `${Math.min(index * 0.04, 0.3)}s` }}
            >
              <div className="bubble">
                <p>{message.content}</p>
                {message.warnings && message.warnings.length > 0 ? (
                  <div className="warning">
                    {message.warnings.map((warning) => (
                      <span key={warning}>{warning}</span>
                    ))}
                  </div>
                ) : null}
                {message.sql ? (
                  <details className="details">
                    <summary>SQL-запит</summary>
                    <pre>
                      <code>{message.sql}</code>
                    </pre>
                  </details>
                ) : null}
                {message.rows && message.columns ? (
                  <details className="details">
                    <summary>Попередній перегляд даних</summary>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            {message.columns.map((column) => (
                              <th key={column}>{column}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {message.rows.slice(0, 25).map((row, rowIndex) => (
                            <tr key={rowIndex}>
                              {row.map((cell, cellIndex) => (
                                <td key={`${rowIndex}-${cellIndex}`}>{String(cell ?? "")}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                ) : null}
              </div>
            </div>
          ))}
          {loading ? (
            <div className="message assistant">
              <div className="bubble thinking">
                <span>Збираю дані...</span>
              </div>
            </div>
          ) : null}
          <div ref={endRef} />
        </div>

        <div className="composer">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Наприклад: Покажи продажі за останні 30 днів по категоріях"
            rows={2}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? "Працюю" : "Надіслати"}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </section>
    </div>
  );
}
