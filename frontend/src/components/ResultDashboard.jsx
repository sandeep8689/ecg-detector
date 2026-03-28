import { useState } from 'react'
import RiskBadge from './RiskBadge'
import ProbabilityChart from './ProbabilityChart'
import RangesPanel from './RangesPanel'
import HeatmapViewer from './HeatmapViewer'
import AbnormalityMeter from './AbnormalityMeter'
import ClinicalInfo from './ClinicalInfo'
import PreprocessingInfo from './PreprocessingInfo'
import DownloadReport from './DownloadReport'

export default function ResultDashboard({ result, onReset }) {
  const [activeTab, setActiveTab] = useState('overview')
  const { prediction, preprocessing, images, filename } = result

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '🏥' },
    { id: 'analysis', label: 'AI Analysis', icon: '🔬' },
    { id: 'ranges', label: 'ECG Ranges', icon: '📊' },
    { id: 'clinical', label: 'Clinical Info', icon: '🩺' },
  ]

  return (
    <div className="py-8 animate-in fade-in duration-500">
      {/* Top bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <button
            onClick={onReset}
            className="flex items-center gap-2 text-slate-400 hover:text-white text-sm font-body transition-colors mb-3 group"
          >
            <svg className="w-4 h-4 group-hover:-translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            New Analysis
          </button>
          <h2 className="font-display text-2xl font-bold text-white">
            Analysis Results
          </h2>
          <p className="text-slate-500 text-sm font-body mt-0.5">{filename}</p>
        </div>

        <DownloadReport result={result} />
      </div>

      {/* Hero result card */}
      <div className="glass-card rounded-2xl p-6 mb-6 relative overflow-hidden">
        {/* Background glow based on risk */}
        <div className={`absolute inset-0 opacity-5 ${
          prediction.risk_level === 'CRITICAL' ? 'bg-red-500' :
          prediction.risk_level === 'HIGH' ? 'bg-orange-500' : 'bg-green-500'
        }`} />

        <div className="relative grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Main detection */}
          <div className="md:col-span-2">
            <div className="flex items-start gap-4">
              <RiskBadge riskLevel={prediction.risk_level} size="lg" />
              <div>
                <p className="text-slate-400 text-xs font-body uppercase tracking-widest mb-1">Detection Result</p>
                <h3 className="font-display text-2xl font-bold text-white mb-1">
                  {prediction.disease}
                </h3>
                <p className="text-slate-300 text-sm font-body leading-relaxed max-w-lg">
                  {prediction.clinical_info.short_summary}
                </p>
              </div>
            </div>
          </div>

          {/* Confidence + abnormality */}
          <div className="flex gap-4 md:justify-end">
            <StatPill label="AI Confidence" value={`${prediction.confidence}%`}
              color={prediction.confidence > 90 ? 'text-green-400' : 'text-yellow-400'} />
            <StatPill label="Abnormalities" value={`${prediction.clinical_info.abnormality_count}/${prediction.clinical_info.total_checks}`}
              color={prediction.clinical_info.abnormality_count === 0 ? 'text-green-400' : 'text-red-400'} />
          </div>
        </div>

        {/* Urgency banner */}
        <div className={`mt-5 rounded-xl px-4 py-3 border text-sm font-body flex items-start gap-2 ${
          prediction.risk_level === 'CRITICAL'
            ? 'bg-red-950/50 border-red-800/40 text-red-300'
            : prediction.risk_level === 'HIGH'
            ? 'bg-orange-950/50 border-orange-800/40 text-orange-300'
            : 'bg-green-950/50 border-green-800/40 text-green-300'
        }`}>
          <span className="text-lg leading-none mt-0.5">
            {prediction.risk_level === 'CRITICAL' ? '🚨' : prediction.risk_level === 'HIGH' ? '⚠️' : '✅'}
          </span>
          <span>{prediction.clinical_info.urgency}</span>
        </div>
      </div>

      {/* Abnormality Meter */}
      <AbnormalityMeter
        count={prediction.clinical_info.abnormality_count}
        total={prediction.clinical_info.total_checks}
        percentage={prediction.clinical_info.abnormality_percentage}
        riskLevel={prediction.risk_level}
        abnormalities={prediction.clinical_info.abnormalities_found}
      />

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-slate-900/60 rounded-xl p-1 border border-slate-800">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-display font-medium transition-all
              ${activeTab === tab.id
                ? 'bg-gradient-to-r from-red-700 to-red-800 text-white shadow-lg'
                : 'text-slate-400 hover:text-slate-200'
              }`}
          >
            <span className="hidden sm:inline">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <OverviewTab prediction={prediction} images={images} preprocessing={preprocessing} />
        )}
        {activeTab === 'analysis' && (
          <AnalysisTab prediction={prediction} images={images} />
        )}
        {activeTab === 'ranges' && (
          <RangesPanel ranges={prediction.ranges_report} riskLevel={prediction.risk_level} />
        )}
        {activeTab === 'clinical' && (
          <ClinicalInfo info={prediction.clinical_info} disease={prediction.disease} />
        )}
      </div>

      {/* Disclaimer */}
      <div className="mt-8 glass-card rounded-xl px-5 py-4 border border-yellow-800/20">
        <p className="text-yellow-600/80 text-xs font-body leading-relaxed text-center">
          ⚠️ <strong>Medical Disclaimer:</strong> {result.disclaimer}
        </p>
      </div>
    </div>
  )
}

function StatPill({ label, value, color }) {
  return (
    <div className="glass-card rounded-xl px-4 py-3 text-center border border-slate-700/50">
      <p className={`font-display text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-slate-500 text-xs font-body mt-0.5">{label}</p>
    </div>
  )
}

function OverviewTab({ prediction, images, preprocessing }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <ProbabilityChart probabilities={prediction.all_probabilities} disease={prediction.disease} />
      <HeatmapViewer images={images} />
      <div className="lg:col-span-2">
        <PreprocessingInfo preprocessing={preprocessing} modelInfo={prediction.model_info} />
      </div>
    </div>
  )
}

function AnalysisTab({ prediction, images }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <HeatmapViewer images={images} expanded />
      <div className="space-y-4">
        <ProbabilityChart probabilities={prediction.all_probabilities} disease={prediction.disease} />
      </div>
    </div>
  )
}
