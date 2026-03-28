import { useState } from 'react'
import axios from 'axios'

export default function DownloadReport({ result }) {
  const [loading, setLoading] = useState(false)
  const [done, setDone] = useState(false)

  const handleDownload = async () => {
    setLoading(true)
    try {
      const payload = {
        disease: result.prediction.disease,
        confidence: result.prediction.confidence,
        risk_level: result.prediction.risk_level,
        clinical_info: result.prediction.clinical_info,
        ranges_report: result.prediction.ranges_report,
        all_probabilities: result.prediction.all_probabilities,
        overlay_base64: result.images?.overlay_base64 || null,
        filename: result.filename,
      }

      const response = await axios.post('/api/report', payload, {
        responseType: 'blob',
        timeout: 30000,
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.download = `ecg_report_${Date.now()}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)

      setDone(true)
      setTimeout(() => setDone(false), 3000)
    } catch (err) {
      console.error('Download failed:', err)
      alert('Report download failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={handleDownload}
      disabled={loading}
      className={`flex items-center gap-2.5 px-5 py-2.5 rounded-xl font-display font-semibold text-sm
        transition-all duration-200 active:scale-95
        ${done
          ? 'bg-green-800/60 border border-green-600/40 text-green-300'
          : 'bg-gradient-to-r from-blue-700 to-blue-800 hover:from-blue-600 hover:to-blue-700 text-white border border-blue-600/40'
        }
        ${loading ? 'opacity-70 cursor-not-allowed' : ''}
      `}
      style={{ boxShadow: '0 0 20px rgba(59,130,246,0.2)' }}
    >
      {loading ? (
        <>
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Generating...
        </>
      ) : done ? (
        <>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Downloaded!
        </>
      ) : (
        <>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download PDF Report
        </>
      )}
    </button>
  )
}
