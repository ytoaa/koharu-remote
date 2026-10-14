'use client'

import { useEffect, useState } from 'react'
import { commands } from '@koharu/bridge/protocol'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { PreferencePage, PreferenceRow } from './PreferenceFields'

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

type Snapshot = { settings: Record<string, unknown> & { comfy_url?: string; options?: Record<string, unknown> } }
export function CustomPreferences({ onOpenTab: _onOpenTab }: { onOpenTab: (tab: 'providers' | 'translation') => void }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [url, setUrl] = useState('')
  const [status, setStatus] = useState('설정을 읽는 중…')
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    void commands.getCustomModels().then(raw => {
      const value = JSON.parse(raw) as Snapshot
      setSnapshot(value); setUrl(value.settings.comfy_url ?? ''); setStatus('')
    }).catch(() => setStatus('설정을 읽지 못했습니다.'))
  }, [])
  async function save(check: boolean) {
    if (!snapshot) return
    setBusy(true)
    try {
      const origin = url.trim()
      const previous = snapshot.settings.options ?? {}
      const options = { ...previous, inpainting_model: 'Qwen-Image-2.1-Turbo',
        detection: { ...record(previous.detection), model: '' },
        ocr: { ...record(previous.ocr), model: '' },
        qwen_inpainting: { ...record(previous.qwen_inpainting), model: 'Qwen-Image-2.1-Turbo', text_encoder: '', vae: '' } }
      const raw = await commands.saveCustomModels(JSON.stringify({ settings: {
        ...snapshot.settings, comfy_url: origin, enabled: Boolean(origin),
        detection_enabled: true, ocr_enabled: true, inpainting_enabled: true, options },
        vision_token: null, inpainting_token: null }))
      setSnapshot(JSON.parse(raw) as Snapshot)
      setStatus(check ? await commands.checkCustomModels() : origin ? '저장 완료. 워크플로우와 모델 옵션은 ComfyUI에서 관리합니다.' : '저장 완료. 내장 모델을 사용합니다.')
    } catch (error) { setStatus(typeof error === 'string' ? error : '서버 연결을 확인하세요.') }
    finally { setBusy(false) }
  }
  return <PreferencePage title='Custom' description='감지·OCR·인페인트를 ComfyUI에 연결합니다. 모델과 처리 옵션은 ComfyUI의 Koharu 워크플로우에서 관리합니다.'>
    <PreferenceRow title='ComfyUI API 주소' description='비우면 내장 모델을 사용합니다.'>
      <Input type='url' aria-label='ComfyUI API 주소' placeholder='http://서버주소:8188' value={url}
        disabled={busy || !snapshot} onChange={e => setUrl(e.currentTarget.value)} />
    </PreferenceRow>
    <div className='flex gap-2'>
      <Button disabled={busy || !snapshot} onClick={() => void save(false)}>저장</Button>
      <Button variant='outline' disabled={busy || !snapshot || !url.trim()} onClick={() => void save(true)}>저장·연결 확인</Button>
    </div>
    <p role='status' className='text-sm'>{status}</p>
  </PreferencePage>
}
