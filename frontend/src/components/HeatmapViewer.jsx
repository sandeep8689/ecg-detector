import { useState } from 'react'

const VIEWS = [
  { id: 'overlay', label: 'AI Overlay', icon: '🔥', key: 'overlay_base64' },
  { id: 'original', label: 'Original', icon: '📷', key: 'original_base64' },
  { id: 'heatmap', label: 'Heatmap', icon: '🌡️', key: 'heatmap_base64' },
]

export default function HeatmapViewer({ images, expanded = false }) {
  const [activeView, setActiveView] = useState('overlay')

  const activeImg = images[VIEWS.find(v => v.id === activeView)?.key]

  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-800">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-display font-semibold text-white">ECG Visual Analysis</h3>
          <p className="text-xs text-slate-500 font-body mt-0.5">
            Red/yellow zones = AI attention areas
          </p>
        </div>
        {/* View toggle */}
        <div className="flex gap-1 bg-slate-900 rounded-lg p-1">
          {VIEWS.map(view => (
            <button
              key={view.id}
              onClick={() => setActiveView(view.id)}
              className={`px-2.5 py-1.5 rounded-md text-xs font-display font-medium transition-all
                ${activeView === view.id
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-500 hover:text-slate-300'
                }`}
            >
              <span className="hidden sm:inline">{view.icon} </span>{view.label}
            </button>
          ))}
        </div>
      </div>

      {/* Image display */}
      <div className={`relative rounded-xl overflow-hidden bg-slate-900 border border-slate-800
        ${expanded ? 'h-80' : 'h-56'}`}>
        {activeImg ? (
          <>
            <img
              src={`data:image/jpeg;base64,${activeImg}`}
              alt={`ECG ${activeView}`}
              className="w-full h-full object-contain transition-opacity duration-300"
            />
            {/* Scan overlay animation on heatmap */}
            {activeView === 'overlay' && (
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-400/60 to-transparent"
                  style={{ animation: 'scanLine 3s linear infinite' }} />
              </div>
            )}
          </>
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-3">
            <svg className="w-12 h-12 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-slate-600 text-sm font-body">Image not available</p>
          </div>
        )}

        {/* View label badge */}
        <div className="absolute top-2 left-2">
          <span className="glass-card text-xs font-mono text-slate-400 px-2 py-1 rounded-lg border border-slate-700">
            {VIEWS.find(v => v.id === activeView)?.icon} {VIEWS.find(v => v.id === activeView)?.label}
          </span>
        </div>
      </div>

      {/* Legend */}
      {activeView === 'overlay' && (
        <div className="mt-3 flex items-center gap-4 text-xs font-body text-slate-500">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-red-500" /> High attention
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-yellow-500" /> Medium attention
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-blue-500" /> Low attention
          </div>
        </div>
      )}

      {activeView === 'overlay' && (
        <p className="mt-2 text-xs text-slate-600 font-body italic">
          Grad-CAM visualization — shows which ECG regions influenced the AI decision
        </p>
      )}
    </div>
  )
}
