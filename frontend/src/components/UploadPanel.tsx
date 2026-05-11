import { useState, useRef, useCallback, useEffect } from 'react'
import { Upload, FileText, CheckCircle, XCircle, Loader, Trash2, X } from 'lucide-react'
import { documents } from '../api/client'

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

type DocItem = { id: string; filename: string; chunk_count: number; size_bytes: number; created_at: string }

type Props = {
  isLoggedIn: boolean
  onAuthClick: () => void
  onClose: () => void
}

export function UploadPanel({ isLoggedIn, onAuthClick, onClose }: Props) {
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [docs, setDocs] = useState<DocItem[]>([])
  const [deleting, setDeleting] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const loadDocs = useCallback(() => {
    if (!isLoggedIn) return
    documents.list(10).then(({ data }) => setDocs(data as DocItem[])).catch(() => {})
  }, [isLoggedIn])

  useEffect(() => { loadDocs() }, [loadDocs])

  const upload = async (file: File) => {
    setSelectedFile(file)
    setStatus('uploading')
    setMessage('')
    setProgress(0)
    const interval = setInterval(() => setProgress(p => Math.min(p + 8, 90)), 300)
    try {
      const { data } = await documents.upload(file)
      clearInterval(interval)
      setProgress(100)
      setStatus('done')
      setMessage(`Ingested ${data.chunk_count} chunks from "${data.filename}"`)
      loadDocs()
    } catch (e: unknown) {
      clearInterval(interval)
      setProgress(0)
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setStatus('error')
      setMessage(msg ?? 'Upload failed — check file type and size (max 50 MB)')
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) upload(file)
  }

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) upload(file)
    e.target.value = ''
  }

  return (
    <div className="border-b border-white/8 bg-[#0d0d0d]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/5">
        <span className="font-mono text-[10px] text-[#555] tracking-widest uppercase">Upload Document</span>
        <button onClick={onClose} className="text-[#444] hover:text-white transition-colors">
          <X size={13} />
        </button>
      </div>

      <div className="px-4 py-3">
        {!isLoggedIn ? (
          /* Not signed in */
          <div className="flex items-center justify-between py-1">
            <span className="font-mono text-xs text-[#555]">Sign in to upload documents</span>
            <button
              onClick={onAuthClick}
              className="font-mono text-[10px] text-crimson hover:text-crimson-light border border-crimson/30 hover:border-crimson/60 px-3 py-1 transition-all"
            >
              Sign in
            </button>
          </div>
        ) : (
          <>
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => status !== 'uploading' && inputRef.current?.click()}
              className={`relative border border-dashed p-4 text-center transition-all ${
                status === 'uploading'
                  ? 'border-white/10 cursor-default'
                  : dragging
                  ? 'border-crimson bg-crimson/5 cursor-copy'
                  : 'border-white/10 hover:border-crimson/40 cursor-pointer'
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                className="hidden"
                onChange={onFileChange}
                accept=".txt,.md,.pdf,.csv,.json,.rst"
              />

              {status === 'uploading' ? (
                <div className="flex items-center gap-3">
                  <Loader size={14} className="text-crimson animate-spin shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-white/70 text-xs truncate">Processing "{selectedFile?.name}"…</p>
                    <div className="mt-1.5 flex items-center gap-2">
                      <div className="flex-1 h-0.5 bg-white/5">
                        <div className="h-full bg-crimson transition-all duration-300" style={{ width: `${progress}%` }} />
                      </div>
                      <span className="font-mono text-[9px] text-[#444] shrink-0">{progress}%</span>
                    </div>
                  </div>
                </div>
              ) : status === 'done' ? (
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <CheckCircle size={14} className="text-green-500 shrink-0" />
                    <span className="text-green-400/80 text-xs truncate">{message}</span>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); setStatus('idle'); setMessage('') }}
                    className="font-mono text-[9px] text-crimson hover:text-crimson-light border border-crimson/30 px-2 py-0.5 shrink-0 transition-colors"
                  >
                    Upload another
                  </button>
                </div>
              ) : status === 'error' ? (
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <XCircle size={14} className="text-crimson shrink-0" />
                    <span className="text-crimson/80 text-xs truncate">{message}</span>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); setStatus('idle'); setMessage('') }}
                    className="font-mono text-[9px] text-crimson hover:text-crimson-light border border-crimson/30 px-2 py-0.5 shrink-0 transition-colors"
                  >
                    Try again
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <Upload size={14} className={dragging ? 'text-crimson' : 'text-[#444]'} />
                  <div className="text-left">
                    <p className="text-white/60 text-xs">Drop a file or click to browse</p>
                    <p className="font-mono text-[9px] text-[#444] mt-0.5">.TXT · .MD · .PDF · .CSV · .JSON · .RST · max 50 MB</p>
                  </div>
                </div>
              )}
            </div>

            {/* Indexed docs list */}
            {docs.length > 0 && (
              <div className="mt-3">
                <p className="font-mono text-[9px] text-[#444] uppercase tracking-widest mb-1.5">
                  Indexed ({docs.length})
                </p>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {docs.map(doc => (
                    <div
                      key={doc.id}
                      className="flex items-center justify-between border border-white/5 px-2.5 py-1.5 hover:border-white/10 transition-all"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <FileText size={10} className="text-[#444] shrink-0" />
                        <span className="text-white/60 text-[11px] truncate">{doc.filename}</span>
                      </div>
                      <div className="flex items-center gap-3 shrink-0 ml-2">
                        <span className="font-mono text-[9px] text-[#333]">{doc.chunk_count} chunks</span>
                        <button
                          onClick={async () => {
                            setDeleting(doc.id)
                            try {
                              await documents.delete(doc.id)
                              setDocs(d => d.filter(x => x.id !== doc.id))
                            } catch { /* ignore */ }
                            finally { setDeleting(null) }
                          }}
                          disabled={deleting === doc.id}
                          className="text-[#333] hover:text-crimson transition-colors disabled:opacity-40"
                        >
                          {deleting === doc.id
                            ? <Loader size={10} className="animate-spin" />
                            : <Trash2 size={10} />}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
