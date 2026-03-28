import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/bmp': ['.bmp'],
  'image/tiff': ['.tiff', '.tif'],
  'application/pdf': ['.pdf'],
  'application/dicom': ['.dcm'],
}

const FORMAT_BADGES = ['JPG', 'PNG', 'PDF', 'DICOM', 'BMP', 'TIFF']

function AnalyzingOverlay({ filename }) {
  return (
    <div className="absolute inset-0 glass-card rounded-2xl flex flex-col items-center justify-center z-20">
      <div className="relative mb-6">
        {[1,2,3].map(i => (
          <div key={i} className="absolute inset-0 rounded-full border border-red-500/30"
            style={{ animation: `pulseRing ${1 + i * 0.4}s ease-out infinite`, animationDelay: `${i * 0.3}s` }} />
        ))}
        <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-red-600 to-red-900 flex items-center justify-center"
          style={{boxShadow:'0 0 30px rgba(239,68,68,0.5)'}}>
          <svg viewBox="0 0 24 24" className="w-10 h-10 animate-beat" fill="none">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="rgba(255,255,255,0.9)"/>
          </svg>
        </div>
      </div>

      <h3 className="font-display text-xl font-bold text-white mb-2">Analyzing ECG...</h3>
      <p className="text-slate-400 text-sm mb-6 font-body">{filename}</p>

      {/* ECG scanning animation */}
      <div className="w-64 h-12 relative overflow-hidden rounded-lg bg-slate-900/50 border border-blue-500/20">
        <svg viewBox="0 0 256 48" className="w-full h-full">
          <path
            className="ecg-line"
            d="M0,24 L40,24 L50,24 L55,8 L60,40 L65,8 L70,40 L75,24 L90,24 L100,24 L110,4 L115,44 L120,4 L125,44 L130,24 L160,24 L170,24 L175,8 L180,40 L185,8 L190,40 L195,24 L220,24 L256,24"
            fill="none" stroke="#3b82f6" strokeWidth="2"
          />
        </svg>
        {/* Scan line */}
        <div className="scan-overlay absolute inset-0" />
      </div>

      <div className="mt-6 flex gap-2">
        {['Preprocessing', 'Running Model', 'Generating Report'].map((step, i) => (
          <div key={step} className="flex items-center gap-1.5 text-xs text-slate-400"
            style={{ opacity: 0.4 + i * 0.3 }}>
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"
              style={{ animationDelay: `${i * 0.3}s` }} />
            {step}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function UploadSection({ setResult, loading, setLoading, error, setError }) {
  const [previewUrl, setPreviewUrl] = useState(null)
  const [uploadedFile, setUploadedFile] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    setUploadedFile(file)
    setError(null)

    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    } else {
      setPreviewUrl(null)
    }
  }, [setError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
    onDropRejected: (fileRejections) => {
      const err = fileRejections[0]?.errors[0]
      if (err?.code === 'file-too-large') setError('File too large. Maximum 50MB.')
      else if (err?.code === 'file-invalid-type') setError('Unsupported format. Use JPG, PNG, PDF, DICOM, BMP, or TIFF.')
      else setError('Invalid file. Please try again.')
    }
  })

  const handleAnalyze = async () => {
    if (!uploadedFile) return
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)

      const response = await axios.post('https://ecg-detector-q67.onrender.com/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000
      })

      setResult(response.data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Analysis failed. Please try again.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleRemove = () => {
    setUploadedFile(null)
    setPreviewUrl(null)
    setError(null)
  }

  return (
    <section className="py-12">
      {/* Hero text */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 bg-red-950/40 border border-red-800/30 rounded-full px-4 py-1.5 mb-6">
          <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
          <span className="text-red-300 text-xs font-body font-medium">AI-Powered Cardiac Screening</span>
        </div>
        <h2 className="font-display text-4xl sm:text-5xl font-bold text-white mb-4 leading-tight">
          Detect Heart Anomalies
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-orange-400">
            in Seconds
          </span>
        </h2>
        <p className="text-slate-400 text-lg max-w-xl mx-auto font-body leading-relaxed">
          Upload any ECG image or PDF report. Our AI model — trained on 10,000+ cardiac scans — delivers clinical-grade analysis instantly.
        </p>
      </div>

      {/* Upload Card */}
      <div className="max-w-2xl mx-auto">
        <div className="relative">
          {loading && uploadedFile && <AnalyzingOverlay filename={uploadedFile.name} />}

          <div
            {...getRootProps()}
            className={`upload-zone relative rounded-2xl border-2 border-dashed p-8 cursor-pointer text-center
              ${isDragActive ? 'dragging border-blue-400' : 'border-slate-700'}
              ${uploadedFile ? 'border-blue-500/50' : ''}
            `}
          >
            <input {...getInputProps()} />

            {!uploadedFile ? (
              <>
                {/* Upload icon */}
                <div className="mx-auto mb-5 w-16 h-16 rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 flex items-center justify-center">
                  <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>

                <p className="text-white font-display font-semibold text-lg mb-1">
                  {isDragActive ? 'Drop your ECG here' : 'Drag & drop ECG file'}
                </p>
                <p className="text-slate-500 text-sm mb-5 font-body">or click to browse</p>

                {/* Format badges */}
                <div className="flex flex-wrap justify-center gap-2">
                  {FORMAT_BADGES.map(fmt => (
                    <span key={fmt} className="px-2.5 py-1 rounded-full text-xs font-mono bg-slate-800 text-slate-400 border border-slate-700">
                      {fmt}
                    </span>
                  ))}
                </div>
                <p className="text-slate-600 text-xs mt-3 font-body">Any size • Blurry images OK • Max 50MB</p>
              </>
            ) : (
              <div className="flex items-center gap-4" onClick={e => e.stopPropagation()}>
                {/* Preview or file icon */}
                <div className="flex-shrink-0 w-20 h-20 rounded-xl overflow-hidden bg-slate-800 border border-slate-700">
                  {previewUrl ? (
                    <img src={previewUrl} alt="ECG preview" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <svg className="w-10 h-10 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                  )}
                </div>

                <div className="flex-1 text-left">
                  <p className="text-white font-display font-semibold truncate">{uploadedFile.name}</p>
                  <p className="text-slate-400 text-sm font-body">
                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB • {uploadedFile.type || 'ECG File'}
                  </p>
                  <div className="flex items-center gap-1.5 mt-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                    <span className="text-green-400 text-xs font-body">Ready to analyze</span>
                  </div>
                </div>

                <button
                  onClick={handleRemove}
                  className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-800 hover:bg-red-900/50 flex items-center justify-center transition-colors"
                >
                  <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mt-3 flex items-center gap-2 text-red-400 text-sm font-body bg-red-950/30 border border-red-800/30 rounded-xl px-4 py-3">
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              {error}
            </div>
          )}

          {/* Analyze button */}
          {uploadedFile && !loading && (
            <button
              onClick={handleAnalyze}
              className="mt-4 w-full py-4 rounded-xl font-display font-bold text-lg text-white
                bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600
                transition-all duration-200 active:scale-[0.98]"
              style={{boxShadow:'0 0 30px rgba(239,68,68,0.3)'}}
            >
              🔍 Analyze ECG
            </button>
          )}
        </div>

        {/* Trust signals */}
        <div className="mt-6 flex flex-wrap justify-center gap-6 text-xs text-slate-500 font-body">
          {['🔒 Secure Analysis', '⚡ Results in ~3 seconds', '🤖 EfficientNet-B4 Model', '📊 96.4% Accuracy'].map(s => (
            <span key={s}>{s}</span>
          ))}
        </div>
      </div>
    </section>
  )
}
