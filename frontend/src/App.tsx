import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Request timeout in milliseconds
const REQUEST_TIMEOUT = 60000;

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  sql?: string;
  columns?: string[];
  rows?: Array<Array<unknown>>;
  warnings?: string[];
  failed?: boolean;
  originalMessage?: string;
};

type ChatResponse = {
  answer: string;
  sql: string | null;
  columns: string[];
  rows: Array<Array<unknown>>;
  warnings: string[];
};

type ErrorResponse = {
  error_code: string;
  message: string;
  details?: Record<string, unknown>;
};

const makeId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function parseErrorResponse(text: string): string {
  try {
    const data = JSON.parse(text);
    if (data.detail) {
      if (typeof data.detail === "string") {
        return data.detail;
      }
      const err = data.detail as ErrorResponse;
      return err.message || "Unknown error";
    }
    return text;
  } catch {
    return text || "Request failed";
  }
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Вітаю! Задавайте питання українською, і я створю SQL-запит для вашої бази даних та поясню результати.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const compactHistory = useMemo(
    () =>
      messages
        .filter((msg) => msg.role === "user" || msg.role === "assistant")
        .filter((msg) => !msg.failed)
        .slice(-6)
        .map((msg) => ({ role: msg.role, content: msg.content })),
    [messages]
  );

  const sendMessageWithContent = useCallback(
    async (messageContent: string) => {
      const trimmed = messageContent.trim();
      if (!trimmed || loading) {
        return;
      }

      const userMessage: ChatMessage = {
        id: makeId(),
        role: "user",
        content: trimmed,
      };

      const nextMessages = [...messages, userMessage];
      setMessages(nextMessages);
      setInput("");
      setLoading(true);
      setError(null);

      // Create abort controller for timeout
      const controller = new AbortController();
      abortControllerRef.current = controller;
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: trimmed, history: compactHistory }),
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const text = await response.text();
          throw new Error(parseErrorResponse(text));
        }

        const data = (await response.json()) as ChatResponse;
        const assistantMessage: ChatMessage = {
          id: makeId(),
          role: "assistant",
          content: data.answer,
          sql: data.sql || undefined,
          columns: data.columns.length > 0 ? data.columns : undefined,
          rows: data.rows.length > 0 ? data.rows : undefined,
          warnings: data.warnings.length > 0 ? data.warnings : undefined,
        };
        setMessages([...nextMessages, assistantMessage]);
      } catch (err) {
        clearTimeout(timeoutId);

        let errorMessage: string;
        if (err instanceof Error) {
          if (err.name === "AbortError") {
            errorMessage = "Request timed out. Please try again.";
          } else {
            errorMessage = err.message;
          }
        } else {
          errorMessage = "Unknown error";
        }

        setError(errorMessage);
        setMessages((prev) => [
          ...prev,
          {
            id: makeId(),
            role: "assistant",
            content:
              "Виникла помилка при обробці запиту. Перевірте підключення та спробуйте ще раз.",
            failed: true,
            originalMessage: trimmed,
          },
        ]);
      } finally {
        setLoading(false);
        abortControllerRef.current = null;
      }
    },
    [messages, loading, compactHistory]
  );

  const sendMessage = useCallback(() => {
    sendMessageWithContent(input);
  }, [input, sendMessageWithContent]);

  const retryMessage = useCallback(
    (originalMessage: string) => {
      // Remove the failed message before retrying
      setMessages((prev) => prev.filter((msg) => msg.originalMessage !== originalMessage));
      sendMessageWithContent(originalMessage);
    },
    [sendMessageWithContent]
  );

  const cancelRequest = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Concord Insight</p>
          <h1>Легкий доступ до даних</h1>
          <p className="subtitle">
            Задавайте питання українською. Система знайде потрібні JOIN-и та поверне результат.
          </p>
        </div>
        <div className="status" aria-label="Service status: online">
          <span className="pulse" aria-hidden="true" />
          <span>LLM + SQL Server</span>
        </div>
      </header>

      <section className="chat" aria-label="Chat interface">
        <div
          className="messages"
          role="log"
          aria-label="Chat messages"
          aria-live="polite"
        >
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`message ${message.role}${message.failed ? " failed" : ""}`}
              style={{ animationDelay: `${Math.min(index * 0.04, 0.3)}s` }}
            >
              <div className="bubble">
                <p>{message.content}</p>
                {message.warnings && message.warnings.length > 0 ? (
                  <div className="warning" role="alert">
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
                    <summary>
                      Результати запиту ({message.rows.length} рядків
                      {message.rows.length >= 25 ? ", показано перші 25" : ""})
                    </summary>
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
                                <td key={`${rowIndex}-${cellIndex}`}>
                                  {String(cell ?? "")}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                ) : null}
                {message.failed && message.originalMessage ? (
                  <button
                    className="retry-button"
                    onClick={() => retryMessage(message.originalMessage!)}
                    aria-label="Retry this message"
                  >
                    Спробувати ще раз
                  </button>
                ) : null}
              </div>
            </div>
          ))}
          {loading ? (
            <div className="message assistant">
              <div className="bubble thinking">
                <span>Обробляю запит...</span>
                <button
                  className="cancel-button"
                  onClick={cancelRequest}
                  aria-label="Cancel request"
                >
                  Скасувати
                </button>
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
            placeholder="Наприклад: покажи продажі за останні 30 днів по категоріях"
            rows={2}
            aria-label="Type your question"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            aria-label={loading ? "Processing request" : "Send message"}
            aria-busy={loading}
          >
            {loading ? "Обробка..." : "Надіслати"}
          </button>
        </div>
        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
      </section>
    </div>
  );
}
