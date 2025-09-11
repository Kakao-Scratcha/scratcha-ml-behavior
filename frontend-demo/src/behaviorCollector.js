// behaviorCollector.js
// 역할: 이벤트 캡처, ROI/좌표 처리, payload 구성, /collect 업로드 (단건/청크), 세션 관리
// React 비의존(순수 JS). App.jsx는 이 모듈을 호출만 한다.

// ---- 환경변수/상수 ----
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const MOVE_FLUSH_MS = 50
const FREE_FLUSH_MS = 120
const MAX_SINGLE_UPLOAD_BYTES = 900_000
const TARGET_CHUNK_BYTES = 300_000
const FETCH_TIMEOUT_MS = 20_000

// ---- 내부 상태 ----
let getHostEl = null
let onRefreshClick = null
let onDebug = null

let runKey = 0
let sessionId = newSessionId()
let t0 = nowMs()

let canvasEl = null
let deviceType = 'unknown'
let sent = false

let events = []
let moveBuf = []
let freeBuf = []
let moveTimer = null
let freeTimer = null
let isDragging = false

// ---- 유틸 ----
function uuid() {
  return (crypto.randomUUID?.() || Math.random().toString(36).slice(2))
}
function newSessionId() { return uuid() }
function nowMs() { return (performance?.now?.() ?? Date.now()) }

function estimateBytes(obj) {
  try { return new TextEncoder().encode(JSON.stringify(obj)).length }
  catch { return (JSON.stringify(obj) || '').length }
}

function* walkDeep(root) {
  const stack = [root]
  while (stack.length) {
    const node = stack.pop()
    if (!node) continue
    if (node.nodeType === 1) { // Element
      yield node
      const children = node.children || []
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i])
      const sr = node.shadowRoot
      if (sr) stack.push(sr)
    } else if (node instanceof ShadowRoot) {
      const children = node.children || []
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i])
    }
  }
}

function getElByRole(role) {
  for (const el of walkDeep(document)) {
    if (el?.matches && el.matches(`[data-role="${role}"]`)) return el
  }
  return null
}
function getRoleFromEl(el) {
  try {
    if (!el || !el.closest) return ''
    const hit = el.closest('[data-role]')
    return hit?.getAttribute?.('data-role') || ''
  } catch { return '' }
}
function elementFromPointDeep(x, y) {
  let el = document.elementFromPoint(x, y)
  let last = null
  while (el && el.shadowRoot && el.shadowRoot.elementFromPoint) {
    const inner = el.shadowRoot.elementFromPoint(x, y)
    if (!inner || inner === last || inner === el) break
    last = el = inner
  }
  return el
}
function getRoleAtPoint(x, y) {
  const el = elementFromPointDeep(x, y)
  return getRoleFromEl(el)
}

function rectOfRole(role) {
  const el = document.querySelector(`[data-role="${role}"]`)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { left: r.left, top: r.top, w: r.width, h: r.height }
}

function buildRoiMap() {
  const roles = [
    'scratcha-container',
    'canvas-container',
    'instruction-area',
    'instruction-container',
    'refresh-button',
    'answer-container',
    'answer-1', 'answer-2', 'answer-3', 'answer-4',
  ]
  const rm = {}
  for (const k of roles) {
    const rr = rectOfRole(k)
    if (rr && rr.w > 0 && rr.h > 0) rm[k] = rr
  }
  delete rm['instruction-container']
  return rm
}

function toNorm(clientX, clientY) {
  const el = canvasEl || getElByRole('canvas-container') || (getHostEl?.() ?? null)
  if (!el) return null
  const r = el.getBoundingClientRect()
  const x_raw = clientX, y_raw = clientY
  const xr = (x_raw - r.left) / Math.max(1, r.width)
  const yr = (y_raw - r.top) / Math.max(1, r.height)
  const oob = (xr < 0 || xr > 1 || yr < 0 || yr > 1) ? 1 : 0
  const x = Math.min(1, Math.max(0, xr))
  const y = Math.min(1, Math.max(0, yr))
  const on_canvas = oob ? 0 : 1
  return { x, y, x_raw, y_raw, on_canvas, oob }
}

// 업로드 직전 슬림 스키마(원본 값 보존, 구조만 슬림화)
function pruneForUpload(fullPayload) {
  const meta = fullPayload?.meta || {};
  const evs = Array.isArray(fullPayload?.events) ? fullPayload.events : [];
  const labelIn = fullPayload?.label || undefined;

  const metaSlim = {
    session_id: meta.session_id,
    viewport: meta.viewport ? { w: Number(meta.viewport.w || 0), h: Number(meta.viewport.h || 0) } :
      { w: window.innerWidth, h: window.innerHeight },
    ts_resolution_ms: 1,
    roi_map: meta.roi_map || {},
  };
  if (meta.device) metaSlim.device = String(meta.device);
  if (meta.dpr != null) metaSlim.dpr = Number(meta.dpr);

  const slimEvents = [];
  for (const e of evs) {
    const t = e?.type;
    if (t === 'moves' || t === 'moves_free') {
      const p = e?.payload || {};
      slimEvents.push({
        type: t,
        payload: {
          base_t: Number(p.base_t || 0),
          dts: (p.dts || []).map(n => Math.max(1, Number(n))),
          xrs: (p.xrs || []).map(Number),
          yrs: (p.yrs || []).map(Number),
        }
      })
      continue
    }
    if (t === 'pointerdown' || t === 'pointerup' || t === 'click') {
      const out = {
        type: t,
        t: Number(e.t || 0),
        x_raw: Number(e.x_raw),
        y_raw: Number(e.y_raw),
      }
      if (t === 'click') {
        if (e.target_role) out.target_role = String(e.target_role)
        if (e.target_answer) out.target_answer = String(e.target_answer)
      }
      slimEvents.push(out)
      continue
    }
  }

  let label = undefined
  if (labelIn && (labelIn.passed != null || labelIn.selectedAnswer || labelIn.error)) {
    label = {}
    if (labelIn.passed != null) label.passed = labelIn.passed ? 1 : 0
    if (labelIn.selectedAnswer) label.selectedAnswer = String(labelIn.selectedAnswer)
    if (labelIn.error) label.error = String(labelIn.error)
  }

  return { meta: metaSlim, events: slimEvents, label }
}

async function fetchJSON(url, body, { timeoutMs = FETCH_TIMEOUT_MS } = {}) {
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort('timeout'), timeoutMs)
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: false,
      signal: ac.signal,
    })
    clearTimeout(timer)
    if (!res.ok) {
      const txt = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${txt}`)
    }
    return await res.json().catch(() => ({}))
  } finally {
    clearTimeout(timer)
  }
}

function sliceEventsByBytes(meta, events, targetBytes) {
  const chunks = []
  let cur = []
  let curBytes = estimateBytes({ meta, events: [] })
  for (const ev of events) {
    const addBytes = estimateBytes(ev)
    if (cur.length > 0 && (curBytes + addBytes) > targetBytes) {
      chunks.push(cur)
      cur = []
      curBytes = estimateBytes({ meta: { session_id: meta.session_id }, events: [] })
    }
    cur.push(ev)
    curBytes += addBytes
  }
  if (cur.length) chunks.push(cur)
  return chunks
}

// ---- 버퍼 플러시 ----
function stopMoveTimer() { if (moveTimer) { clearInterval(moveTimer); moveTimer = null } }
function startMoveTimer() { if (!moveTimer) moveTimer = setInterval(() => flushMoves(), MOVE_FLUSH_MS) }
function stopFreeMoveTimer() { if (freeTimer) { clearInterval(freeTimer); freeTimer = null } }
function startFreeMoveTimer() { if (!freeTimer) freeTimer = setInterval(() => flushFreeMoves(), FREE_FLUSH_MS) }

function flushMoves() {
  const buf = moveBuf
  if (!buf.length) return
  const base_t = Math.round(buf[0].t - t0)
  const xrs = [], yrs = [], dts = []
  for (let i = 0; i < buf.length; i++) {
    const cur = buf[i]
    const prevT = i === 0 ? buf[0].t : buf[i - 1].t
    xrs.push(cur.x_raw)
    yrs.push(cur.y_raw)
    dts.push(Math.max(1, Math.round(cur.t - prevT)))
  }
  events.push({ type: 'moves', payload: { base_t, dts, xrs, yrs } })
  moveBuf = []
}
function flushFreeMoves() {
  const buf = freeBuf
  if (!buf.length) return
  const base_t = Math.round(buf[0].t - t0)
  const xrs = [], yrs = [], dts = []
  for (let i = 0; i < buf.length; i++) {
    const cur = buf[i]
    const prevT = i === 0 ? buf[0].t : buf[i - 1].t
    xrs.push(cur.x_raw)
    yrs.push(cur.y_raw)
    dts.push(Math.max(1, Math.round(cur.t - prevT)))
  }
  events.push({ type: 'moves_free', payload: { base_t, dts, xrs, yrs } })
  freeBuf = []
}

// ---- 세션 관리 ----
export function resetSession() {
  try { stopMoveTimer() } catch {}
  try { stopFreeMoveTimer() } catch {}
  events = []
  moveBuf = []
  freeBuf = []
  sent = false
  isDragging = false
  t0 = nowMs()
  sessionId = newSessionId()
  runKey++
  // canvasEl은 다음 MutationObserver에서 다시 발견됨
  return sessionId
}
export function getSessionId() { return sessionId }

// ---- 캡처 바인딩 ----
function onPointerDown(e) {
  deviceType = String(e.pointerType || 'unknown')
  const n = toNorm(e.clientX, e.clientY)
  if (!n) return
  const t = nowMs()
  isDragging = true
  startMoveTimer()
  moveBuf = []
  freeBuf = []
  events.push({ t: Math.round(t - t0), type: 'pointerdown', x_raw: n.x_raw, y_raw: n.y_raw })
  moveBuf.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
}
function onPointerMove(e) {
  if (e.pointerType) deviceType = String(e.pointerType)
  const n = toNorm(e.clientX, e.clientY)
  if (!n) return
  const t = nowMs()
  if (isDragging) {
    moveBuf.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
  } else {
    startFreeMoveTimer()
    freeBuf.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
  }
}
function onPointerUp(e) {
  const n = toNorm(e.clientX, e.clientY)
  if (!n) { isDragging = false; stopMoveTimer(); flushMoves(); return }
  const t = nowMs()
  if (isDragging) {
    moveBuf.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
  }
  isDragging = false
  stopMoveTimer()
  flushMoves()
  freeBuf = []
  events.push({ t: Math.round(t - t0), type: 'pointerup', x_raw: n.x_raw, y_raw: n.y_raw })
}
function onPointerCancel() {
  isDragging = false
  stopMoveTimer()
  flushMoves()
}
function onClick(e) {
  const role = getRoleAtPoint(e.clientX, e.clientY)
  const answerText = (e.target?.getAttribute?.('data-answer') || '').trim() || null
  const n = toNorm(e.clientX, e.clientY)
  if (!n) return
  const t = Math.round(nowMs() - t0)
  events.push({
    t, type: 'click', x_raw: n.x_raw, y_raw: n.y_raw,
    target_role: String(role || ''), target_answer: String(answerText || '')
  })
  // 위젯의 새로고침 버튼을 UI에서 누른 경우
  if (role === 'refresh-button') {
    e.preventDefault?.()
    if (typeof onRefreshClick === 'function') onRefreshClick()
  }
}

let mutationObs = null
export function startCapture() {
  const rootEl = (getHostEl?.() ?? document)
  // pointer
  const optWin = { passive: true, capture: true }
  window.addEventListener('pointerdown', onPointerDown, optWin)
  window.addEventListener('pointermove', onPointerMove, optWin)
  window.addEventListener('pointerup', onPointerUp, optWin)
  window.addEventListener('pointercancel', onPointerCancel, optWin)
  // click (capture)
  const optCap = { capture: true, passive: true }
  rootEl.addEventListener?.('click', onClick, optCap)

  // canvas-container 자동 추적
  mutationObs = new MutationObserver(() => {
    const latest = getElByRole('canvas-container')
    if (latest && latest !== canvasEl) canvasEl = latest
  })
  mutationObs.observe(document, { childList: true, subtree: true, attributes: true })
}
export function stopCapture() {
  const rootEl = (getHostEl?.() ?? document)
  rootEl.removeEventListener?.('click', onClick, { capture: true })
  window.removeEventListener('pointerdown', onPointerDown, { capture: true })
  window.removeEventListener('pointermove', onPointerMove, { capture: true })
  window.removeEventListener('pointerup', onPointerUp, { capture: true })
  window.removeEventListener('pointercancel', onPointerCancel, { capture: true })
  if (mutationObs) { mutationObs.disconnect(); mutationObs = null }
  stopMoveTimer(); stopFreeMoveTimer()
}

// ---- 초기화/옵션 ----
export function init(opts = {}) {
  getHostEl = opts.getHostEl || null
  onRefreshClick = opts.onRefreshClick || null
  onDebug = opts.onDebug || null
  // 세션 초기화(앱 시작 시 1회)
  resetSession()
}

// ---- 업로드/검증 ----
function buildMeta(roiMap) {
  return {
    session_id: sessionId,
    device: deviceType,
    viewport: { w: window.innerWidth, h: window.innerHeight },
    dpr: window.devicePixelRatio || 1,
    ts_resolution_ms: 1,
    roi_map: roiMap || {},
  }
}

function debugSet(obj) {
  try { onDebug && onDebug(obj) } catch {}
}

export async function postCollect(label) {
  if (sent) return { verification: null, bytes: 0, evCount: 0, roiKeys: [] }
  try {
    stopMoveTimer(); stopFreeMoveTimer()
    flushMoves(); flushFreeMoves()

    const roiMap = buildRoiMap()
    const metaFull = buildMeta(roiMap)
    const fullPayload = { meta: metaFull, events, label }
    const slim = pruneForUpload(fullPayload)

    let verification = null
    const singleBytes = estimateBytes({ meta: slim.meta, events: slim.events, label: slim.label })
    const roiKeys = Object.keys(slim.meta.roi_map || {})

    if (singleBytes <= MAX_SINGLE_UPLOAD_BYTES) {
      const res = await fetchJSON(`${API_URL}/collect`, { meta: slim.meta, events: slim.events, label: slim.label })
      verification = res?.verification ?? null
    } else {
      const { meta, events: evs } = slim
      const chunks = sliceEventsByBytes(meta, evs, TARGET_CHUNK_BYTES)
      for (let i = 0; i < chunks.length; i++) {
        const isFirst = (i === 0)
        const partMeta = isFirst ? meta : { session_id: meta.session_id }
        await fetchJSON(`${API_URL}/collect_chunk`, {
          meta: partMeta,
          events: chunks[i],
          label: null,
          part_index: i,
          total_parts: chunks.length,
        })
      }
      const fin = await fetchJSON(`${API_URL}/collect_finalize`, {
        meta: { session_id: meta.session_id },
        events: [],
        label: slim.label ?? label ?? undefined,
      })
      verification = fin?.verification ?? null
    }

    sent = true
    const evCount = events.length
    events = []
    moveBuf = []
    freeBuf = []

    const dbg = { lastBytes: singleBytes, lastEvents: evCount, roiKeys, verification }
    debugSet(dbg)
    return { verification, bytes: singleBytes, evCount, roiKeys }
  } catch (e) {
    const verification = { ok: false, error: (e?.message || String(e) || 'unknown') }
    debugSet({ verification })
    return { verification, bytes: 0, evCount: 0, roiKeys: [] }
  }
}
