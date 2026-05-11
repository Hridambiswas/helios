# Helios Frontend

React 18 + TypeScript + Vite single-page application for the Helios AI platform.

## Stack

| Package | Purpose |
|---------|---------|
| React 18 | UI framework |
| TypeScript | Type safety |
| Vite | Build tool |
| Tailwind CSS | Styling |
| Framer Motion | Animations |
| axios | API requests |
| react-markdown | Markdown answer rendering |
| lucide-react | Icons |

## Project Structure

```
src/
├── api/
│   └── client.ts          # Axios instance, types, API helpers
├── components/
│   ├── QueryInterface.tsx  # Main query UI — input, pipeline progress, results
│   ├── AuthModal.tsx       # Login / register modal
│   ├── Hero.tsx            # Landing hero section
│   ├── Navbar.tsx          # Top navigation
│   ├── UploadSection.tsx   # Document upload
│   ├── HistorySection.tsx  # Query history
│   ├── PipelineSection.tsx # Pipeline visualization
│   └── SplashScreen.tsx    # Initial loading screen
├── hooks/                  # Custom React hooks
├── styles/                 # Global CSS
├── App.tsx                 # Root component + routing
└── main.tsx                # Entry point
```

## Development

```bash
cd frontend
npm install
cp .env.example .env.local   # set VITE_API_URL=http://localhost:8000
npm run dev
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VITE_API_URL` | Backend base URL |
| `VITE_GUEST_QUERY_LIMIT` | Free queries before login prompt (-1 = unlimited) |

## Key Components

### QueryInterface
Handles the full query lifecycle:
1. User types query → submits
2. WebSocket connection (if logged in) streams pipeline events
3. Pipeline progress bar animates through steps
4. Result rendered with markdown, citations, follow-up questions

### Follow-up Questions
After each answer, 2 contextual follow-up questions appear as clickable chips. Clicking one immediately runs the follow-up query.

### Sources Tab
Shows two sections:
- **Web Sources** — DuckDuckGo results with clickable title links
- **Knowledge Base** — local indexed documents with similarity scores

## Build

```bash
npm run build    # outputs to dist/
npm run preview  # serve the production build locally
```

Vercel auto-deploys on every push to `main`.
