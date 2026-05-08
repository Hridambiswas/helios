import { useState, useRef } from 'react'
import { Upload, FileText, CheckCircle, XCircle, Loader } from 'lucide-react'
import { documents } from '../api/client'

export function UploadSection({ isLoggedIn }: { isLoggedIn: boolean }) {
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  if (!isLoggedIn) return null

  const upload = async (file: File) => {
    setStatus('uploading')
    setMessage('')
    try {
      const { data } = await documents.upload(file)
      setStatus('done')
      setMessage(`Ingested ${data.chunk_count} chunks from "${data.filename}"`)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setStatus('error')
      setMessage(msg ?? 'Upload failed')
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
  }

  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-baseline gap-6 mb-12">
          <span className="section-number" style={{ fontSize: 'clamp(40px,6vw,72px)' }}>05</span>
          <div>
            <div className="hr-red w-16 mb-2" />
            <h2 className="font-mono text-xs tracking-[0.3em] uppercase text-crimson">Knowledge Base</h2>
            <p className="text-white text-2xl font-light mt-1">Ingest Documents</p>
          </div>
        </div>

        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative border border-dashed p-12 text-center cursor-pointer transition-all ${
            dragging ? 'border-crimson bg-crimson/5' : 'border-white/10 hover:border-crimson/40'
          }`}
        >
          <input ref={inputRef} type="file" className="hidden" onChange={onFileChange}
            accept=".txt,.md,.pdf,.csv,.json,.rst" />

          <div className="flex flex-col items-center gap-4">
            {status === 'uploading' ? (
              <Loader size={32} className="text-crimson animate-spin" />
            ) : status === 'done' ? (
              <CheckCircle size={32} className="text-green-500" />
            ) : status === 'error' ? (
              <XCircle size={32} className="text-crimson" />
            ) : (
              <div className={`p-4 border ${dragging ? 'border-crimson bg-crimson/10' : 'border-white/10'} transition-all`}>
                <Upload size={24} className={dragging ? 'text-crimson' : 'text-[#555]'} />
              </div>
            )}

            <div>
              <p className="text-white/70 text-sm">
                {status === 'uploading' ? 'Chunking, embedding, indexing...' :
                 status === 'done' ? message :
                 status === 'error' ? message :
                 'Drop a file here or click to browse'}
              </p>
              {status === 'idle' && (
                <p className="font-mono text-[10px] text-[#444] mt-1 tracking-wider">
                  .TXT · .MD · .PDF · .CSV · .JSON · .RST · max 50 MB
                </p>
              )}
            </div>

            {(status === 'done' || status === 'error') && (
              <button
                onClick={e => { e.stopPropagation(); setStatus('idle'); setMessage('') }}
                className="font-mono text-[10px] text-crimson hover:text-crimson-light tracking-wider uppercase border border-crimson/30 px-3 py-1 transition-colors"
              >
                Upload Another
              </button>
            )}
          </div>
        </div>

        {/* Supported types */}
        <div className="mt-4 flex gap-6 justify-center">
          {['.TXT', '.MD', '.PDF', '.CSV', '.JSON', '.RST'].map(ext => (
            <div key={ext} className="flex items-center gap-1.5 text-[10px] font-mono text-[#444]">
              <FileText size={10} />
              {ext}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
