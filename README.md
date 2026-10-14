# Koharu Remote

공식 [Koharu (koharu-rs/koharu)](https://github.com/koharu-rs/koharu)의 모델 호출 경계에 패치를 적용하고 ComfyUI/DGX 서버를 연결하는 독립적인 확장 프로젝트입니다. 공식 Koharu 프로젝트가 배포하거나 보증하는 빌드가 아닙니다.

## 원본 코드와 라이선스

- **공식 Koharu 원본:** [koharu-rs/koharu](https://github.com/koharu-rs/koharu), 현재 기준 버전 `0.83.5`, commit `52bb4f55e2266f72c1523562a583c827ec191ceb`. 고정 정보는 [upstream.json](upstream.json)에 있습니다. 원본의 **MIT OR Apache-2.0** 라이선스와 저작권 고지를 [LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)에 보존합니다.
- **Hi-SAM 추론 코드:** [ymy-k/Hi-SAM](https://github.com/ymy-k/Hi-SAM)의 Apache-2.0 코드 일부를 포함합니다. 원본 저작권, 전체 라이선스, 로컬 수정 내역은 [LICENSE](dgx/vision/_text_sam/LICENSE)와 [NOTICE](dgx/vision/_text_sam/NOTICE)에 있습니다.
- **RORem 서비스 코드 및 외부 모델:** 가져온 서비스의 원본·revision은 [provenance.json](dgx/rorem/provenance.json)에 기록합니다. 전체 출처와 모델별 조건은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고하세요. 모델 가중치에는 각 모델 배포자의 별도 라이선스가 적용됩니다.

현재 패치 버전은 `0.9.0`입니다. ComfyUI 설치·업데이트·사용 방법은 [comfyui/README.md](comfyui/README.md)를 참고하세요. SAM 글자 마스크, 중복 마스크 보존, 국소 색 보정과 한국어 옵션은 서버와 워크플로우 업데이트로 적용합니다. 아래의 별도 중계 컨테이너 안내는 선택형 구성입니다.

별도 Koharu Remote Windows 앱과 DGX 모델 서비스를 만드는 저장소입니다. 이 버전은 **활성 인페인트를 RORem-4S에서 Qwen Image 2.1 Turbo로 대체**합니다. 감지·Hayai OCR과 기존 번역용 vLLM은 계속 사용합니다. 공식 Koharu 실행파일은 수정하지 않습니다.

| 활성 서비스 | 기능 | 포트 |
|---|---|---:|
| Vision | RF-DETR 감지 + Hayai OCR v2, 한 프로세스·한 추론 큐 | 8766 |
| Qwen | Qwen Image 2.1 Turbo 글자 제거·배경 복원 | 8765 |
| 기존 vLLM | Gemma 번역, 외부에서 관리 | 8000 `/v1` |

Qwen 적용 시 기존 RORem 컨테이너는 중지하고 복구용으로 보관합니다. 동시에 두 인페인트 서비스를 실행하거나 앱에 모델 선택 메뉴를 제공하지 않습니다. 기존 인페인트 URL·인증 설정을 이어받습니다.

## Settings → Custom

서버 모델 사용과 감지/OCR/인페인트 사용을 단계별로 켭니다. 기본 주소는 loopback이며, Windows에서 DGX에 직접 연결할 경우 기존 개인 서버 주소를 사용합니다. 응답 대기 시간은 기본 600초, 10–1800초 범위입니다. 서버 장애는 오류로 표시하고 내장 모델로 자동 전환하지 않습니다.

Custom 설정은 `~/.koharu/config.toml`의 `[custom-models]`에 저장합니다. 인증 토큰은 현재 native 실행 메모리에만 보관합니다. 인증 없는 서버는 토큰을 비워 둡니다. 토큰 앞뒤 공백·반복 Bearer 접두사를 정규화하고 개행·제어문자는 거부합니다. 처리 중에는 설정 저장을 거부하며, 저장 후 다음 요청부터 반영합니다.

| 옵션 | 기본값 / 범위 |
|---|---|
| 감지 임계값 | 글자 0.25, 효과음 0.20, 말풍선·패널 0.50; 공식 설정에서 호환값 가져오기 |
| Hayai 최대 출력 토큰 | 256 / 1–2048 |
| Hayai 반복 패널티 | 1.0 / 1.0–2.0 |
| Hayai 빔 수 | 1(greedy) / 1–4 |
| Hayai 길이 패널티·조기 종료 | 1.0·켜짐 / 빔 검색에만 적용 |
| 인페인트 | Qwen Image 2.1 Turbo, 영역별 복원 기본 켜짐 |
| Qwen 생성 스케줄 | 체크포인트에 저장된 8스텝·CFG 1, 고정 |
| Qwen 복원 지시 | 글자 제거·주변 배경/선화/망점 복원 지시, 편집 가능 |
| 시드 | 0 기본. -1이면 각 HTTP 요청마다 랜덤, 0 이상은 고정 |
| 처리 해상도 | 긴 변 기본 2048, **256–4096** 선택 가능 |
| 추론 마스크 여백 | 기본 0 / 0–32 모델 공간 px |
| 컨텍스트 캐시 | 기본 켜짐. 바꾸면 결과와 처리 시간이 달라질 수 있음 |

Qwen 2.1의 실제 API에는 `mask_image`·`strength`가 없습니다. 원본과 별도 흑백 마스크를 두 참조 이미지로 전달하고, 흰 영역만 지우도록 지시합니다. 입력은 긴 변 상한 안에서 비율을 유지해 축소하고 32픽셀 배수의 정사각형으로 패딩합니다. 결과에서 패딩을 제거하고 원본 크기로 복원한 뒤 **원래 마스크 안에만 합성**합니다. 모델 공간 마스크 확대가 최종 마스크 밖을 덮어쓰지 않습니다.

4096은 선택 상한이며 입력이 작으면 강제로 4096으로 확대하지 않습니다. Qwen의 공식 예시는 2048 정사각형 편집이며, 4096은 사용자 요청으로 연 범위입니다. GB10에서 4096 추론 성공·메모리 사용·만화 선화/망점 보존은 아직 검증하지 않았습니다. OOM이면 명확한 오류로 반환하며 해상도를 몰래 낮추지 않습니다.

Hayai의 전처리·코하루 기본값·고정 모델 revision과 사용하지 않는 SigLIP pooled-head 제외는 유지합니다. 출력 한도 도달 또는 긴 반복 루프 결과는 SourceText로 저장하지 않고 오류로 반환합니다. 짧은 효과음은 허용합니다. 영역별 인페인트에서 시드 -1은 영역마다 별도 난수를 사용합니다.

## DGX: Qwen으로 교체

먼저 이전 Windows 앱 폴더를 보관합니다. 새 버전은 다른 폴더에 압축 해제합니다. 기존 RORem 컨테이너·이미지·모델 파일을 삭제하지 않습니다. `.env.rorem`으로 이미 이관했다면 adopt를 다시 실행하지 않습니다. 이관하지 않았다면 실행 중인 기존 RORem에 `python3 manage.py rorem adopt`를 한 번 실행합니다.

새 Qwen 체크포인트는 아래 고정 revision에서 약 **32.4 GB 가중치**와 구성 파일을 받습니다. 모델·Docker 이미지·캐시를 위한 추가 디스크 공간이 필요합니다.

```bash
cd ~/koharu-remote
git pull --ff-only
cd dgx
python3 manage.py vision build
python3 manage.py vision up
python3 manage.py qwen init \
  --model-dir /srv/koharu/models/qwen-image-2.1-turbo \
  --cache-dir /srv/koharu/cache/qwen
python3 manage.py qwen build
python3 manage.py qwen download
python3 manage.py qwen check
python3 manage.py qwen up
python3 manage.py qwen wait
python3 manage.py all health
```

`init/build/download/check`는 기존 RORem을 계속 실행한 상태에서 준비합니다. `qwen up`만 기존 RORem의 idle/컨테이너 identity를 확인하고 중지한 뒤 Qwen을 8765에 시작합니다. Docker 시작 명령 자체가 실패하면 원래 RORem을 다시 시작합니다. Qwen preload/워밍업 실패 또는 실제 품질 불만은 아래 rollback으로 복구합니다.

Qwen 모델은 `Qwen/Qwen-Image-2.1-Turbo@fc13ba1986fcb905854dc8de2f52eee395f49b5b`에 고정합니다. 16개 필수 파일의 공식 SHA/크기를 검증합니다. Diffusers는 `916a5791ed5f29d966200d2eee53f06862ec4434`에 고정하며 Turbo의 저장된 8스텝 sigma grid를 검증합니다. CUDA BF16으로 한 번 preload하고 256px 합성 입력으로 실제 8스텝 워밍업한 후 ready가 됩니다. 서빙 시 모델은 read-only mount와 오프라인 로딩을 사용하며 요청마다 다시 로드하지 않습니다.

Qwen의 새 이미지 안에서만 의존성을 설치하고 NVIDIA torch/torchvision/torchaudio를 설치 전후 비교·constraints로 보호합니다. 호스트 torch/CUDA와 번역 vLLM은 변경하지 않습니다. 기존 RORem 모델 경로와 Qwen의 새 모델/캐시 경로가 겹치면 준비를 거부합니다. serving과 명시적 다운로드 단계를 분리해 서빙 컨테이너가 모델 파일을 쓰지 않게 합니다.

관리 명령은 `manage.py vision …`, `manage.py qwen …`, `manage.py all …`입니다. all은 **현재 활성 인페인트 하나만** 관리합니다. Qwen 준비만 했을 때는 RORem, Qwen 적용 후에는 Qwen, rollback 후에는 다시 RORem을 관리합니다. Qwen 로그: `python3 manage.py qwen logs`. 수신 주소 변경: `python3 manage.py qwen bind 0.0.0.0` 후 `qwen up`. init은 기존 RORem의 수신 주소·인증을 이어받습니다.

### Qwen 품질이 별로일 때 rollback

1. Windows 앱에서 실행 중인 작업을 끝내고 Qwen 버전 앱을 종료합니다.
2. DGX에서 다음을 실행합니다. 원래 RORem 컨테이너·이미지 identity가 일치할 때만 복구합니다.

```bash
cd ~/koharu-remote/dgx
python3 manage.py qwen rollback
python3 manage.py all health
```

3. 새 Windows 앱 폴더에서 PowerShell로 아래 스크립트를 실행하고, 보관한 이전 Windows 앱을 다시 엽니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\rollback-custom-settings.ps1
```

새 앱은 Custom 설정을 처음 읽기 전에 `~/.koharu/config.before-qwen-0.6.0.toml`을 한 번 만들고 덮어쓰지 않습니다. 복원 스크립트는 이전 **Custom 섹션만** 복원하고 현재 번역/제공자 설정은 유지합니다. 복원 직전 전체 설정도 별도 파일로 보관합니다. 모델 가중치·캐시·이미지는 삭제하지 않습니다. Qwen 앱이 RORem 서버에 조용히 연결되는 fallback은 없습니다.

## 기존 메뉴 그대로 처리

프로젝트의 기존 감지/OCR/번역/인페인팅 메뉴를 사용합니다. 범위·작업 큐·scene/mask·commit은 기존 Koharu가 담당합니다. 번역 제공자는 기존 OpenAI-compatible 설정과 `:8000/v1`을 사용합니다. 공식 앱 실행파일·공식 모델 설정을 변경하지 않습니다.

## 최신 Actions 빌드

같은 PR/브랜치의 새 실행이 시작되면 `concurrency.cancel-in-progress=true`로 이전 빌드를 취소합니다. 최초 적용 시 concurrency가 없었던 이전 실행도 같은 workflow/branch/event의 더 오래된 실행에 한해 취소합니다. 다른 PR·브랜치·더 최신 실행은 취소하지 않습니다.

Actions의 **Koharu CI and source**는 PR/main 변경에서 검사와 소스 artifact를 생성합니다. **Windows 앱도 빌드**를 체크한 수동 실행만 Windows artifact를 만듭니다. 서버 업데이트에는 이미 실행 중인 클라이언트를 그대로 사용합니다. Windows artifact가 필요한 경우 전체 ZIP을 새 폴더에 풀고 `bin/koharu-remote.exe` 또는 `start-remote.ps1`로 실행합니다. 소스 artifact에는 모델 가중치·개인 .env·state를 넣지 않습니다. Windows/DGX 실제 검증 상태는 `VALIDATION.md`를 확인합니다.

**Windows artifact repackage (manual)**은 지정한 성공 run의 artifact를 재포장하는 수동 복구 기능입니다. 예전 run 번호를 기본값으로 넣거나 push에서 자동 실행하지 않습니다. 성공한 main 검사 뒤에는 닫힌 PR의 브랜치 중 현재 tip 전체가 main에 포함된 항목만 정리합니다. 기본/활성 DGX 브랜치, 열린 PR에 쓰이는 브랜치, 보호 브랜치, 고유 커밋이 남은 브랜치는 유지합니다. 완료된 Actions 로그와 artifact는 삭제하지 않습니다.

### 최초 clone: 헤드리스 DGX

공개 저장소이므로 소스를 읽고 clone하는 데 GitHub 로그인이 필요하지 않습니다.

```bash
cd "$HOME"
git clone https://github.com/ytoaa/koharu-remote.git
```

기존 배포에서 이전한다면 새 checkout으로 소스를 받습니다. 기존 개인 `.env`·서비스별 설정·ComfyUI `config.json`·적용 프로필은 별도로 보존하고 필요한 새 checkout 위치에 복사하세요. 실제 서버 주소와 모델/캐시 경로는 개인 설정에 유지하며 커밋하지 않습니다. 같은 서비스의 컨테이너를 중복 실행하지 마세요. ComfyUI의 custom-node symlink는 새 checkout을 가리키도록 업데이트한 뒤 ComfyUI를 재시작합니다.

이미 clone했다면 저장소에서 `git pull --ff-only`만 실행합니다. Git checkout에는 배포용 `SHA256SUMS`가 없습니다. Actions 소스 artifact를 사용한다면 먼저 `python3 tools/verify_sums.py .`로 검증합니다.

### 감지·OCR 첫 설치 또는 업데이트

이미 실행 중인 vision의 `.env`는 그대로 재사용합니다. 인페인트만 교체하는 경우 vision을 다시 build/up할 필요가 없습니다. 처음 설치하는 경우:

```bash
(
set -eu
cd "$HOME/koharu-remote/dgx"
python3 manage.py vision prepare --download-detector --download-ocr
python3 manage.py vision check
python3 manage.py vision build
python3 manage.py vision up
)
```

preload 완료 뒤:

```bash
python3 manage.py vision health
python3 manage.py vision doctor
python3 manage.py vision logs
```

vision build는 기존 로컬 `local/koharu-rorem:0.2.3`을 바탕으로 새 이미지에서만 패키지를 설치합니다. NVIDIA torch/torchvision/torchaudio는 constraints와 설치 전후 비교로 보호합니다. 기본 `VISION_AUTH_REQUIRED=false`, `VISION_API_TOKEN=`이므로 token 없이 사용합니다. 인증을 켜려면 `.env`의 flag와 token을 설정하고 `python3 manage.py vision up`으로 반영합니다. `vision prepare`는 다운로드 옵션 없이 실행하면 기존 감지와 Hayai 파일을 검증합니다. 기존 설치 업데이트는 `vision prepare --download-ocr → vision build → vision up`입니다. 기존 `.env`에는 새 Hayai 경로만 추가하며 주소와 토큰은 보존합니다.

신규 설치 기본 경로 예시 (실제 경로는 `dgx/.env`에서 지정): `/srv/koharu/models/koharu-layout-rfdetr-seg-2xl-1152`. 공식 Koharu가 사용하는 동일 checkpoint repository/revision과 safetensors·loader·config SHA256을 고정합니다. `prepare --download-detector`는 **이 신규 감지 모델만** 다운로드합니다. 이미 존재하는 다른 내용의 파일은 덮어쓰지 않습니다. vision은 기존 `/srv/koharu/models/rorem4s`를 mount하지 않습니다. RORem은 최초 이관 때 가져온 기존 경로를 read-only로 mount합니다. Hayai 위치는 `/srv/koharu/models/hayai-ocr-v2`이며 `prepare --download-ocr`로 622,502,784-byte 가중치와 고정 코드·토크나이저를 검증해 받습니다. 감지/Hayai는 read-only mount, 캐시는 `/srv/koharu/cache/vision`입니다.

DGX Linux의 host network에서 감지·OCR 공통 Vision endpoint를 제공합니다. Uvicorn의 수신 주소는 `.env`의 `VISION_BIND_ADDRESS`로 지정하며 기본값은 `127.0.0.1`입니다. Qwen의 DGX 공개 주소는 `.env.qwen`의 `QWEN_BIND_ADDRESS`입니다. 두 서비스 모두 `0.0.0.0` 또는 접속에 사용할 DGX 인터페이스 IP로 직접 수신하도록 설정할 수 있고, 수신 주소와 토큰 인증 여부는 독립적으로 설정합니다. `0.0.0.0`은 Tailscale뿐 아니라 다른 IPv4 인터페이스에서도 받습니다. 단일 worker와 inference gate를 사용하며, 실패나 클라이언트 취소 후에도 CUDA thread가 끝나기 전까지 gate를 놓지 않습니다. OOM 시 오류를 반환하며 다른 GPU 프로세스를 종료하거나 서버 모델을 unload하지 않습니다.

서버 추론은 신규 구현이므로 GPU/메모리/검출 결과 비교를 별도 검증해야 합니다. Detector는 원본 이미지를 RF-DETR의 1152×1152 모델 입력 경계에 전달하고, 원본 좌표의 bbox와 cropped segmentation mask를 반환합니다. 타일/crop 추론과 추가 page 축소는 없습니다. GPU 커널·dtype에 따른 정확한 수치 일치는 아직 확인하지 않았습니다. Qwen 인페인트의 모델 입력은 선택한 긴 변 상한 안에 맞춰 정사각형으로 패딩하며, 결과는 원본 크기로 복원합니다.

Hayai는 코하루 고정 모델 `JustANormalTinkerer/hayai-ocr-v2@4a4ce477c9a8841f208b94e1d9ed5c0938965e05`를 사용합니다. SHA256을 검증한 배포 모델 코드에 패키지 Config import 연결, 오프라인 vision config, 코하루 dtype 선택을 적용합니다. Python/Transformers 런타임이며 코하루의 Rust 구현과 픽셀·커널 수치까지 같다고 보장하지 않습니다. 고급 옵션을 저장하면 각 OCR 요청에 적용하고, 설정 복원은 코하루 기본값을 다시 적용합니다. OCR에 vLLM을 호출하지 않으며 기존 vLLM은 번역용으로 유지됩니다. health의 `ocr_model_loaded`, `ocr_model`, `ocr_revision`, `ocr_warmup_complete`로 실제 로드 상태를 확인합니다.

0.3.0의 개발용 Custom OCR 옵션 스키마는 제거했습니다. 그 버전에서 Custom 모델 옵션을 이미 저장했다면 앱을 종료하고 `config.toml`의 `[custom-models.options]` 아래 OCR 하위 테이블만 제거한 뒤 0.4.0을 실행합니다. 나머지 주소·옵션과 공식 번역 설정은 유지합니다.

인페인트 이미지·마스크 업로드 제한은 파일마다 기본 128 MiB입니다. `.env.qwen`의 `QWEN_MAX_UPLOAD_BYTES`와 `QWEN_MAX_PIXELS`로 조절합니다. 초과 업로드는 구조화된 HTTP 413으로 반환하고, 정상 결과는 원본 이미지 크기의 PNG인지 확인합니다.

## 자동 upstream 대응

`watch.yml`은 매일 공식 stable release를 확인합니다. 새 release의 **정확한 commit**을 `upstream.json`에 고정하는 후보 branch/PR을 만들고, `windows.yml`을 명시적으로 호출해 **기존 패치 적용 → Python 테스트 → Compose config 검증 → 소스 checksum artifact**를 수행합니다. GITHUB_TOKEN으로 만든 PR이 별도 workflow를 자동 시작하지 않는 문제는 reusable workflow 호출로 처리합니다. dependency drift 검사·native 회귀 테스트·명령 bindings 생성·Windows 빌드는 후보 branch를 선택한 수동 실행에서 **Windows 앱도 빌드**를 체크해야 실행합니다.

정상 적용 가능한 업데이트는 자동으로 후보 검사와 소스 artifact까지 생성됩니다. **패치 충돌·API 변경·테스트 실패·의존성 버전 변경은 자동 승격하지 않습니다.** main과 기존 정상 빌드는 유지하며 maintainer가 후보를 수정/검증 후 merge합니다. GPU 기능 검증 없이 자동 merge하거나 자동 배포하지 않습니다. 같은 후보 branch가 이미 있으면 매일 같은 검사를 반복하지 않습니다. 실제 upstream 승격 전에는 별도 Windows 빌드와 필요한 실기 검증을 수행하세요. scheduled workflow는 default branch에서만 실행됩니다. GitHub Settings → Actions → General에서 Actions의 PR 생성을 허용해야 합니다. 서버 secrets, code signing secrets는 이 workflow에 필요하지 않습니다.

관리 저장소는 [ytoaa/koharu-remote](https://github.com/ytoaa/koharu-remote)이며 기본 branch는 `main`입니다. `main` push와 PR에서 offline 검사와 소스 배포를 실행합니다. Windows 빌드는 입력 `build_windows=true`로 명시한 수동/reusable 호출에서만 실행합니다. 경로 기반 판정은 ComfyUI 코드·워크플로우·서버 회귀 검사도 서버 변경으로 분류합니다. 저장소의 Actions 설정에서 **Allow GitHub Actions to create and approve pull requests**를 켜야 watcher가 후보 PR을 생성할 수 있습니다.

### 수동 Windows build (maintainer용)

Windows Actions runner가 Rust stable MSVC/C++ 빌드 환경을 제공합니다. 사용하는 사람의 PC에는 이 개발 환경을 설치할 필요가 없습니다.

```powershell
python tools/checkout.py build/upstream
python tools/apply.py build/upstream
python tools/lock.py build/upstream
cd build/upstream
bun install --frozen-lockfile
cargo test --locked -p koharu-pipeline remote::tests --lib
cargo run --locked -p koharu-app --bin generate
$config = (Resolve-Path ../../tauri.remote.json).Path
cargo tauri build --no-bundle --config $config -- --locked
cd ../..
python tools/package_windows.py build/upstream
```

소스 수정은 9개 기존 파일 + 3개 신규 모듈입니다. generated `protocol.ts`와 `Cargo.lock`을 손으로 수정하지 않습니다. bindings는 Rust command 등록에서 생성하고, Cargo의 workspace lock 갱신 후 dependency 버전/source drift를 검사합니다. 패치 생성은 maintainer가 `tools/make_patch.py`로 검토하여 수행하며 업데이트 Action은 충돌 패치를 임의 재생성하지 않습니다.

## ComfyUI Turbo bridge (0.7.0)

The tested ComfyUI Qwen 2.1 Turbo INT8 graph can serve the existing `/v1/inpaint` API on port 8765. The bridge owns no GPU runtime: it calls the already installed ComfyUI on 8188 and never modifies its venv, packages, launch script, models or SageAttention settings. External vLLM remains separate. The bridge uses one artwork reference, the checkpoint's eight sigmas plus terminal zero, Euler, CFG 1 and optional Qwen prefix KV cache. It fits the supplied crop to its aspect ratio without square letterboxing, restores the original crop dimensions, then composites only inside the original binary mask. Comfy instruction editing does not use inference-mask dilation; keep that option at 0. Existing app region tiling remains active.

Keep ComfyUI running with the user's existing script (`--listen 0.0.0.0 --use-sage-attention --disable-mmap --preview-method none`), then:

```bash
cd ~/koharu-remote
# Use the reviewed feature branch until its PR is merged.
git switch feat/hayai-advanced-options-random-seed
git pull --ff-only
cd dgx
python3 manage.py qwen backend comfyui
python3 manage.py qwen build
python3 manage.py qwen up
python3 manage.py qwen wait
```

`backend comfyui` first waits for/validates the current service's idle boundary and stops only that service. Build once for the new CPU-only HTTP bridge. Docker uses `host.docker.internal:8188` with Linux host-gateway; `qwen backend comfyui --url http://<DGX-IP>:8188` overrides this if needed. ComfyUI models/caches remain in `/opt/ComfyUI` and its existing runtime. Startup submits a small eight-step warmup before marking ready. The bridge may wait behind existing ComfyUI queue jobs. It never calls global `/interrupt` or `/free`, and retains its API gate until its own timed-out job reaches a known terminal state. Bridge output images appear under ComfyUI's `output/koharu_bridge...` for inspection.

Defaults are `qwen_image_2.1_turbo_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors`, and `qwen_image_2.1_vae_bf16.safetensors`. The existing Windows 0.6 Qwen API contract can use these defaults; use the 0.7 Windows candidate for editable model names. In Custom settings, enter the diffusion filename, optional encoder filename and optional VAE filename. Blank encoder/VAE retains the currently configured server values. The logical name `Qwen-Image-2.1-Turbo` uses the server's selected diffusion file. Save and check connection to explicitly select and warm up changed filenames via `/v1/models/select`; subsequent inference reuses ComfyUI's caches. Only compatible Qwen 2.1 **Turbo** weights are supported by this fixed eight-step adapter. Other model families need an appropriate loader/sampler adapter, rather than renaming a file. Model changes need no app rebuild or bridge-image build.

Switch back to the retained Diffusers weights/cache without deleting any model:

```bash
python3 manage.py qwen backend diffusers
python3 manage.py qwen build
python3 manage.py qwen up
python3 manage.py qwen wait
```

### Editable detector/OCR identities

Custom settings now accept detector and OCR model IDs rather than enforcing release-specific IDs/revisions in the app. These names must match `/health` (`model` / `ocr_model`); a mismatch is an explicit error. Names do not magically replace weights in a resident Vision process. For compatible RF-DETR/Hayai weights, prepare a separate local model bundle and switch its bind-mount path without rebuilding:

```bash
python3 manage.py vision models --ocr-dir /absolute/path/to/hayai-compatible-bundle
# Or: python3 manage.py vision models --detector-dir /absolute/path/to/layout-compatible-bundle
python3 manage.py vision up
python3 manage.py vision health
```

Each alternative bundle needs `model-files.json` with `repository`, `revision`, and a `files` map of filename to SHA256. Hayai uses the same six files as `dgx/vision/hayai_files.py`; its adapter code and vision-config hashes remain fixed. RF-DETR uses the same three files as `dgx/vision/model_files.py`, retains the pinned `load_model.py`, and `inference_config.json` must declare the actual checkpoint SHA256 and RF-DETR 1.7.0. Weights/config/tokenizer can change within those loader contracts; new architectures or adapter code require implementation changes. Vision preloads on restart, then the app verifies the entered ID. One initial server/app update is necessary to add these controls; later compatible model replacements do not require builds.

Validation: offline HTTP/graph/selection/mask-composite tests pass; the user separately measured the ComfyUI UI workflow at 1248x1248, eight steps, 22.83s. The bridge itself has not yet been run on DGX or through the Windows GUI. This UI measurement is not bridge acceptance evidence.


## 현재 ComfyUI 통합

기존 ComfyUI 프로세스에 감지·OCR·Turbo 인페인트 API를 설치하는 방법은 [comfyui/README.md](comfyui/README.md)를 참고하세요. 별도 중계 컨테이너 없이 Custom의 ComfyUI API 주소 하나로 연결합니다. 사용자는 DGX에서 세 단계 실행과 인페인트 국소 색 보정·중복 마스크 보존의 효과를 확인했습니다. OCR 반복/토큰 제한과 감지 누락은 이미지에 따라 계속 확인할 품질 과제입니다. 모델/품질 설정은 ComfyUI 그래프에서 조절하고 메뉴에서 적용합니다. 위의 예전 컨테이너 안내를 현재 설치 절차와 섞어 사용하지 마세요.
