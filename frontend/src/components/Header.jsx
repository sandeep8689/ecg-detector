export default function Header() {
  return (
    <header className="relative py-8 px-4 sm:px-6 lg:px-8 mb-4">
      {/* ECG line decoration */}
      <div className="absolute bottom-0 left-0 right-0 h-px overflow-hidden">
        <svg viewBox="0 0 1200 20" className="w-full h-5" preserveAspectRatio="none">
          <path
            className="ecg-line"
            d="M0,10 L200,10 L220,10 L230,2 L240,18 L250,2 L260,18 L270,10 L290,10 L310,10 L320,1 L330,19 L340,1 L350,19 L360,10 L380,10 L400,10 L500,10 L520,10 L530,2 L540,18 L550,2 L560,18 L570,10 L590,10 L700,10 L720,10 L730,2 L740,18 L750,2 L760,18 L770,10 L790,10 L1200,10"
            fill="none"
            stroke="rgba(239,68,68,0.4)"
            strokeWidth="1.5"
          />
        </svg>
      </div>

      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="pulse-ring absolute inset-0 rounded-full bg-red-500/20" />
            <div className="relative w-12 h-12 bg-gradient-to-br from-red-600 to-red-800 rounded-xl flex items-center justify-center shadow-lg" style={{boxShadow:'0 0 20px rgba(239,68,68,0.4)'}}>
              <svg viewBox="0 0 24 24" className="w-7 h-7 animate-beat" fill="none">
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="rgba(255,255,255,0.9)" stroke="white" strokeWidth="0.5"/>
              </svg>
            </div>
          </div>
          <div>
            <h1 className="font-display text-xl font-bold text-white tracking-tight">
              Cardio<span className="text-red-400">Scan</span> AI
            </h1>
            <p className="text-xs text-slate-400 font-body">ECG Heart Anomaly Detector</p>
          </div>
        </div>

        {/* Stats strip */}
        <div className="flex items-center gap-6 text-center">
          {[
            { val: "96.4%", label: "Accuracy" },
            { val: "4", label: "Conditions" },
            { val: "< 3s", label: "Analysis" },
          ].map(({ val, label }) => (
            <div key={label} className="flex flex-col">
              <span className="font-display font-bold text-lg text-red-400 glow-text">{val}</span>
              <span className="text-xs text-slate-500 font-body">{label}</span>
            </div>
          ))}
        </div>

        {/* Model badge */}
        <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-slate-300 font-mono">EfficientNet-B4</span>
        </div>
      </div>
    </header>
  )
}
