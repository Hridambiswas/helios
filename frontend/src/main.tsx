import { StrictMode, Component } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles/globals.css'

// Catch any crash and display the error so we can see what's wrong
class RootBoundary extends Component<
  { children: React.ReactNode },
  { error: string | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props)
    this.state = { error: null }
  }
  static getDerivedStateFromError(e: Error) {
    return { error: e?.message ?? String(e) }
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          position: 'fixed', inset: 0,
          background: '#fff', color: '#c00',
          padding: 32, fontFamily: 'monospace',
          fontSize: 14, overflowY: 'auto', zIndex: 99999,
        }}>
          <strong>React crash — please copy this and share:</strong>
          <pre style={{ marginTop: 16, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {this.state.error}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RootBoundary>
      <App />
    </RootBoundary>
  </StrictMode>
)
