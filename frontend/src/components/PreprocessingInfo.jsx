export default function PreprocessingInfo({ preprocessing, modelInfo }) {
  const qualityColor =
    preprocessing.quality_score > 70 ? '#22c55e' :
    preprocessing.quality_score > 40 ? '#f97316' : '#ef4444'

  const qualityLabel =
    preprocessing.quality_score > 70 ? 'Good' :
    preprocessing.quality_score > 40 ? 'Fair' : 'Poor'

  const fileTypeIcons = {
    jpeg: '🖼️', png: '🖼️', pdf: '📄', dicom: '🏥', bmp: '🖼️', tiff: '🖼️'
  }

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-display font-semibold text-white">Image Quality & Processing</h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-green-400 font-mono">Auto-Enhanced</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
        {/* Quality score */}
        <div className="glass-card rounded-xl p-3 border border-slate-700 text-center">
          <p className="font-display text-2xl font-bold" style={{ color: qualityColor }}>
            {preprocessing.quality_score}
          </p>
          <p className="text-xs text-slate-500 font-body">Quality Score</p>
          <span className="text-xs font-body mt-1 inline-block px-2 py-0.5 rounded-full"
            style={{ background: `${qualityColor}20`, color: qualityColor }}>
            {qualityLabel}
          </span>
        </div>

        {/* File type */}
        <div className="glass-card rounded-xl p-3 border border-slate-700 text-center">
          <p className="text-2xl">{fileTypeIcons[preprocessing.file_type] || '📁'}</p>
          <p className="font-display font-bold text-white uppercase text-sm">{preprocessing.file_type}</p>
          <p className="text-xs text-slate-500 font-body">Format</p>
        </div>

        {/* Blur status */}
        <div className="glass-card rounded-xl p-3 border border-slate-700 text-center">
          <p className="text-2xl">{preprocessing.is_blurry ? '🌫️' : '✨'}</p>
          <p className="font-display font-bold text-white text-sm">
            {preprocessing.is_blurry ? 'Enhanced' : 'Clear'}
          </p>
          <p className="text-xs text-slate-500 font-body">Sharpness</p>
        </div>

        {/* Original size */}
        <div className="glass-card rounded-xl p-3 border border-slate-700 text-center">
          <p className="font-display font-bold text-white text-sm">
            {preprocessing.original_size?.[1]}×{preprocessing.original_size?.[0]}
          </p>
          <p className="text-xs text-slate-500 font-body">Original Size</p>
          <p className="text-xs text-blue-400 font-mono mt-1">→ 224×224</p>
        </div>
      </div>

      {/* Issues handled */}
      {preprocessing.issues_handled?.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-slate-500 font-body uppercase tracking-wider mb-2">Issues Auto-Handled:</p>
          <div className="flex flex-wrap gap-2">
            {preprocessing.issues_handled.map(issue => (
              <span key={issue} className="px-2.5 py-1 rounded-full text-xs font-body
                bg-blue-950/40 border border-blue-800/30 text-blue-300">
                ✓ {issue}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Processing steps */}
      <div>
        <p className="text-xs text-slate-500 font-body uppercase tracking-wider mb-2">Processing Pipeline:</p>
        <div className="flex flex-wrap items-center gap-1 text-xs font-mono text-slate-500">
          {['Load & Detect', '→', 'Denoise', '→', 'CLAHE Contrast', '→', 'Auto-Rotate', '→', 'Resize 224×224', '→', 'Normalize', '→', 'Model Input'].map((step, i) => (
            <span key={i} className={step === '→' ? 'text-slate-700' : 'bg-slate-900 px-2 py-0.5 rounded'}>
              {step}
            </span>
          ))}
        </div>
      </div>

      {/* Model info */}
      <div className="mt-4 pt-4 border-t border-slate-800 flex flex-wrap gap-4 text-xs font-body text-slate-500">
        <span>🤖 <strong className="text-slate-400">{modelInfo?.name}</strong></span>
        <span>📊 Trained on <strong className="text-slate-400">{modelInfo?.dataset}</strong></span>
        <span>🎯 Val Accuracy: <strong className="text-green-400">{modelInfo?.accuracy}</strong></span>
      </div>
    </div>
  )
}
