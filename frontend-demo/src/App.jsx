import { useEffect, useRef, useState } from 'react'
import { ScratchaWidget } from 'scratcha-sdk'
import './App.css'
import * as collector from './behaviorCollector'

const SCRATCHA_API_KEY = import.meta.env.VITE_SCRATCHA_API_KEY ?? ''
const SCRATCHA_ENDPOINT = import.meta.env.VITE_SCRATCHA_ENDPOINT ?? 'https://api.scratcha.cloud'

function App() {
  // freeze-only: 정답/에러 후 위젯 숨김(자동 다음 퍼즐 차단)
  const [runKey] = useState(0)
  const [frozen, setFrozen] = useState(false)
  const [debug, setDebug] = useState(null)
  const hostRef = useRef(null)

  // ── 콘솔 로그: 판정결과, 봇일 확률, 임계치, 속도, 시간, OOB비율(캔버스) ──
  const showVerificationLog = (verification) => {
    try {
      if (!verification) { console.warn('[verify] no verification payload'); return }
      if (verification.ok === false && verification.error) {
        console.error('[verify] inference failed:', verification.error)
        console.log('raw:', verification)
        return
      }
      const verdictKo =
        verification.verdict === 'human' ? '사람' :
        verification.verdict === 'bot'   ? '봇'   :
        String(verification.verdict || '알수없음')

      const probPct = typeof verification.bot_prob === 'number'
        ? (verification.bot_prob * 100).toFixed(1) : 'N/A'
      const thr = verification.threshold != null ? verification.threshold : 'N/A'

      const stats = verification?.stats || {}
      const speedMean = (stats?.speed_mean != null) ? stats.speed_mean.toFixed(4) : 'N/A'
      const oobCanvas = (typeof stats.oob_rate_canvas === 'number') ? stats.oob_rate_canvas : null
      const oobPct = (oobCanvas == null) ? 'N/A' : (oobCanvas * 100).toFixed(1)

      console.group('[Scratcha verification]')
      console.log(`판정결과: ${verdictKo}`)
      console.log(`봇일 확률: ${probPct}%`)
      console.log(`임계치: ${thr}`)
      console.log(`속도(평균): ${speedMean}`)
      console.log(`OOB비율(canvas): ${oobPct}%`)
      console.groupEnd()
    } catch (e) {
      console.error('verification log error:', e)
    }
  }

  // 성공/실패 → 수집 후 freeze
  const handleSuccess = async (result) => {
    const selectedAnswer = result?.selectedAnswer ?? result?.answer ?? null
    const { verification, bytes, evCount, roiKeys } =
      await collector.postCollect({ passed: 1, selectedAnswer })
    setDebug({ lastBytes: bytes, lastEvents: evCount, roiKeys, verification })
    showVerificationLog(verification)
    setFrozen(true)
  }

  const handleError = async (error) => {
    const msg = (error?.message ?? error)?.toString?.() ?? 'unknown'
    const { verification, bytes, evCount, roiKeys } =
      await collector.postCollect({ passed: 0, error: msg })
    setDebug({ lastBytes: bytes, lastEvents: evCount, roiKeys, verification })
    showVerificationLog(verification)
    setFrozen(true)
  }

  useEffect(() => {
    collector.init({
      getHostEl: () => hostRef.current,
      onRefreshClick: () => { setFrozen(true) }, // 위젯 내부 새로고침도 freeze
      onDebug: (d) => setDebug(d),
    })
    collector.startCapture()
    return () => { collector.stopCapture() }
  }, [])

  return (
    <>
      <div data-role="scratcha-container" ref={hostRef} style={{ display: 'inline-block' }}>
        {!frozen && (
          <ScratchaWidget
            key={runKey}
            apiKey={SCRATCHA_API_KEY}
            endpoint={SCRATCHA_ENDPOINT}
            mode="normal"
            onSuccess={handleSuccess}
            onError={handleError}
          />
        )}
      </div>
    </>
  )
}

export default App
