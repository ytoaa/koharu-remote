# 0.6.0 — Replace RORem with Qwen Image 2.1 Turbo

- 활성 인페인트를 Qwen 2.1 Turbo로 대체. 기존 endpoint 8765·인증을 유지하고 RORem은 중지 상태로 복구용 보존. 앱의 모델 선택 메뉴 없이 Qwen 옵션만 제공.
- 고정 Qwen checkpoint/16개 파일 hash와 크기, 고정 Diffusers commit, 저장된 8스텝 schedule을 검증. CUDA BF16 preload·워밍업·상주, read-only/offline serving.
- 원본+별도 마스크 참조 이미지와 편집 지시를 사용. 비율 유지 축소·정사각형 패딩·원본 크기 복원·원래 마스크 합성. RORem 전용 옵션을 Qwen에 전달하지 않음.
- Qwen 긴 변 선택 상한 4096, 기본 2048. 요청별 -1 랜덤·고정 시드와 컨텍스트 캐시 옵션. OCR 고급 옵션은 0.5 후보 변경을 포함.
- idle/identity 검사 후 교체, Docker 시작 실패 시 원래 서버 재기동, 명시적 서버 rollback과 Windows Custom 설정 백업/복원.
- 새 Actions 실행 시 같은 PR/브랜치의 이전 빌드 취소. 최초 적용 전 실행도 scope/순서를 검사해 취소.
- GB10 Qwen 실제 추론·품질·4096 메모리 검증은 아직 미실행.

# 0.5.0 — Editable Hayai advanced options and random inpainting seed

- 코하루 Hayai 설정은 최초 기본값으로 유지하고, Custom의 고급 OCR 옵션에서 출력 토큰·반복 패널티·빔 수·길이 패널티·빔 검색 조기 종료를 편집·저장.
- 앱에서 저장한 OCR 옵션을 각 HTTP 요청으로 전달. 서버에서 엄격한 자료형·범위 검사 후 상주 모델에 적용하며 요청 간 설정을 공유하지 않음.
- 출력 한도에서 끊긴 OCR 결과 검증은 실제 요청 한도를 사용. 모델 pin·전처리와 0.4.1 pooler 수정 유지.
- 인페인팅 시드 `-1`은 요청마다 새 난수 시드, `0` 이상은 고정 시드. 기존 기본값 `0` 유지.
- 시드 필드를 signed 정수로 바꾸고, UI에 `-1`의 의미 표시. 서버에 OCR 옵션·랜덤 시드 capability 추가.
- 새 Windows 0.5.0 앱과 Vision 재빌드·RORem update 필요. 모델 파일과 기존 vLLM 변경 없음.

# DGX Vision 0.4.1 — Skip unused Hayai pooled-output head

- 사용자 GB10 로그에서 Hayai 워밍업 중 SigLIP2 pooled-output BF16 MultiheadAttention 실패 확인. 감지는 로딩·워밍업 완료, Hayai 실패로 health 503 유지.
- Hayai에서 쓰지 않는 pooled-output 헤드 실행만 제외. checkpoint 엄격 검증과 코하루 입력·BF16 선택·256-token greedy 생성·반복 패널티는 그대로 유지.
- 실제 checkpoint 테스트에 FP32/BF16 CPU 확인과 불필요한 헤드 실행 금지 hook 추가.
- Vision image 0.4.1 재빌드·재기동 필요. 모델 파일과 기존 Windows 0.4.0 앱은 그대로 사용.

# 0.4.0 — Hayai OCR and one management entry point

- 감지와 Hayai OCR v2를 Vision 단일 프로세스에서 시작 시 함께 preload/warmup. 양쪽 준비 전 health는 503.
- Koharu 0.83.5의 Hayai 모델 revision, 전처리와 생성 기본값 고정. 반복 패널티·출력 길이를 임의로 조정하지 않음.
- 출력 한도에서 끊김 또는 긴 반복을 생성 후 검증. 비정상 SourceText가 번역으로 넘어가는 것을 차단.
- Custom OCR 모델/프롬프트/온도 입력 제거. 앱의 OCR 요청과 health에서 고정 Hayai 모델·revision 확인.
- `manage.py all check/up/health/logs/stop`으로 모델 서비스 관리. 감지·OCR에 별도 Python 실행 불필요.
- `vision prepare --download-ocr` 추가, 파일 SHA256 검사와 read-only mount. 기존 detector 파일·개인 설정 보존.
- 기존 vLLM은 번역용으로 유지. Vision image 재빌드와 새 Windows 앱 필요. DGX GPU 실측은 아직 미실행.
- 0.3.0 Custom OCR 하위 옵션 테이블은 제거: 이미 저장한 개발 버전 옵션은 README의 업데이트 절차 적용.

# 0.3.0 — Custom model options and startup warmup

- 인증 토큰 비저장 검사가 공개 OCR 옵션 `max_tokens`를 오인하던 문제 수정. 인증 필드와 실제 세션 토큰 값을 검사.
- Windows 네이티브 테스트 명령을 각각 별도 단계로 실행하여 앞선 실패가 뒤의 성공에 가려지지 않게 수정. 실패 시 앱 빌드·artifact 업로드로 진행하지 않음.
- 최초 Windows 빌드와 Rust 캐시 저장 시간을 함께 고려해 작업 제한을 150분에서 210분으로 조정. 이전 후보의 빌드 성공과 네이티브 테스트 통과 여부를 VALIDATION.md에서 구분.

- Custom에서 감지의 클래스별 임계값, OCR의 모델·프롬프트·출력 토큰·온도, RORem-4S의 프롬프트·네거티브 프롬프트·스텝·강도·가이던스·시드·처리 해상도·추론 마스크 여백을 관리.
- 최초 모델 옵션은 현재 공식 설정의 호환되는 임계값·RORem 프롬프트와 공식 처리 해상도를 가져옴. 저장 후에는 Custom에 독립적으로 유지. 4S 스텝/가이던스는 모델에 맞춰 4/1로 유지하며 로컬 mixed 모델의 값을 복사하지 않음.
- 공식 설정 가져오기·모델 기본값 복원, 고급 옵션 접기, 저장 후 연결 확인 추가. Custom에서 번역 제공자/서버 주소와 번역 모델/프롬프트 설정으로 바로 이동.
- Remote 인페인팅은 기본적으로 기존 Koharu의 단색 영역 채우기·연결 마스크별 분할·주변 문맥·마스크 합성 경로를 사용. 전체 페이지 처리도 선택 가능. 모델 파일과 read-only mount 유지.
- 서버 OCR/RORem 옵션을 요청에 명시적으로 전달하고 양쪽에서 범위 검사. 옵션 미지원 서버는 연결 확인 때 업데이트 필요 메시지를 표시하여 조용히 무시되지 않게 함.
- 감지 모델 시작 시 1280×1536 합성 입력으로 전처리/1152 모델 추론/마스크 복원을 두 번 워밍업하고 CUDA 완료를 기다린 뒤 ready 처리. 실패하면 미준비 상태. 워밍업 첫/반복 시간, 실제 감지 predict/encode 시간 기록. CUDA 커널 캐시는 기존 vision 캐시 볼륨에 보존.
- 모델 공간의 마스크 확대는 추론에만 적용하고 최종 합성은 원래 요청 마스크로 제한. 축소 과정에서 매우 얇은 선택 영역이 전부 사라지는 경우에는 면적 커버리지로 빈 마스크를 방지.
- Windows 소스·서버 API 옵션이 함께 바뀌므로 새 앱 사용 전 vision 재빌드와 RORem 코드 업데이트 필요. 기존 vLLM, 호스트 torch/CUDA와 모델 가중치는 변경하지 않음.

# 0.2.1 — Large PNG uploads and accurate HTTP errors

- 사용자 DGX 로그의 `/v1/inpaint` HTTP 413을 확인하여, RORem 파일별 업로드 제한을 32 MiB에서 기본 128 MiB로 확대. `ROREM_MAX_UPLOAD_BYTES`로 조정하고 `/health`에서 적용값 확인.
- 초과 업로드에 파일 종류와 제한을 포함한 구조화된 413 응답 및 로그 추가. 모델·픽셀 제한·추론/복원 정책은 유지.
- Windows 클라이언트가 HTTP 오류를 PNG Content-Type 검사보다 먼저 처리. 413/인증/검증/busy 오류와 알려진 서버 오류 code를 표시하고 임의 응답 본문·토큰은 표시하지 않음.
- 정상 인페인팅 PNG 응답 상한을 256 MiB로 조정하고 원본 크기 검사 유지.

# DGX direct bind configuration (Windows patch 0.2.0 unchanged)

- 감지·OCR의 `VISION_BIND_ADDRESS`와 RORem의 `BIND_ADDRESS`를 `manage.py vision bind IP` / `manage.py rorem bind IP`로 설정. `0.0.0.0` 또는 Tailscale IP에서 직접 수신 가능.
- 수신 주소와 인증 flag를 독립적으로 설정. bind 명령은 기존 개인 설정·모델 경로·캐시·토큰을 보존.
- 기존 vision 이미지에도 Compose command/healthcheck 설정으로 적용하여 의존성 설치·재빌드 없이 변경 가능.
- 구체적인 IP의 health 확인 경로를 수정하고, RORem 교체 전 idle 검사에 실행 중인 컨테이너의 실제 기존 바인딩을 사용.

# DGX management revision unified-1 (Windows patch 0.2.0 unchanged)

- RORem 전체 소스를 `dgx/rorem`으로 통합. 기존 inference/backend/모델 검증 코드의 SHA256과 고정 원본을 기록.
- 공통 관리 명령을 `manage.py vision …` / `manage.py rorem …`으로 구분. 이전 bare vision 명령과 외부 RORem source patch 도구 제거.
- `rorem adopt`로 Docker의 실제 모델·캐시 경로, 개인 설정과 검증 이미지를 최초 이관. 예전 버전 폴더의 경로 없이 이후 관리 가능.
- `rorem build/update`는 기존 이미지의 의존성을 유지하고 코드만 변경. 기존 Compose 정체성과 read-only 모델을 유지하고 busy/컨테이너 변경/이미지 변경을 검사.
- adopt 뒤 컨테이너를 먼저 삭제하려면 `rorem remove → build → up → wait` 지원. 기존 vision/vLLM을 관리 대상에 포함하지 않음.
- DGX-only 변경에도 offline 검사·Compose config·소스 패키징을 실행하고 Windows 재빌드를 생략. 네이티브 소스와 Windows build step 변경, 수동/upstream 후보 실행은 전체 빌드 유지.

# 0.2.0 — Stage selection and optional local authentication

- Custom에 감지·OCR·인페인팅의 개별 서버 사용 선택 추가. 끈 단계는 기존 내장 모델을 사용하고 켠 단계만 연결 확인합니다.
- 토큰 없이 설정을 저장하고 HTTP 요청할 수 있게 변경. 빈 토큰이면 Authorization 헤더를 생략합니다. 이미 설정한 세션 토큰을 지우는 버튼 추가.
- vision 인증은 `VISION_AUTH_REQUIRED=false`가 기본이며, 인증을 켠 경우에만 token을 요구합니다. 기존 token은 그대로 유지할 수 있습니다.
- 기존 RORem 0.2.3의 인증 소스만 검증/백업/수정하는 `dgx/patch_rorem_auth.py` 추가. 모델 추론과 의존성은 변경하지 않으며, 실제 컨테이너 반영은 기존 `manage.py update` 경로를 사용합니다.
- 인증 해제와 인증 유지 양쪽의 실제 HTTP handler, 단계별 선택/무헤더 요청, 기존 세션 token 유지/제거, 소스 무결성 검사에 대한 회귀 검사 추가.
- Windows 빌드의 Rust/Tauri 의존성 캐시 추가. 배포 artifact는 배포 폴더만 업로드하며 내부 ZIP을 중복 업로드하지 않습니다.

# 0.1.0 — Separate minimum-patch approach

- Koharu Settings에 Custom 탭 추가. 이유: 외부 Companion에서 상태/lock을 추적하며 처리하는 복잡성을 줄이고 기존 job/scene/commit 경로를 사용.
- 감지/OCR/인페인팅의 모델 추론 호출만 HTTP로 전환. Detection region/관계, OCR 사용자 SourceText 보호, inpainting mask/cleanup 처리 유지.
- 처리 중 설정 저장 차단 및 idle 시 모델 연결 캐시 갱신. 이유: 저장 전 연결이나 내장 모델이 다음 요청에 남는 문제 방지.
- token 세션 메모리 유지, 반복 Bearer/공백 정규화, 빈 값·제어문자 거부. 저장/상태 응답에는 token 원문 없음.
- 신규 detector resident 서비스와 기존 vLLM 이미지 입력 OCR gateway 추가. 기존 RORem/vLLM 및 host 패키지는 변경하지 않음.
- detector 체크포인트/loader/config를 commit·SHA256으로 고정. NVIDIA torch constraints·설치 전후 비교·doctor metadata 적용.
- Actions로 공식 stable release 감시, immutable pin 후보 PR, 자동 patch/test/build. 실패는 main에 반영하지 않으며 충돌은 수동 검토.
- 생성된 bindings/lock을 공식 도구로 갱신. 내장 공식 updater는 별도 빌드 config에서 endpoint를 비움.

현재 Companion/runtime-adapter 저장소의 버전 업이 아닌 별도 프로젝트입니다. 현재 자체 패치 버전은 `0.5.0`, 기반 Koharu는 `0.83.5`입니다. 이후 bugfix는 PATCH, 기능 추가는 MINOR, 안정 API의 호환 파괴는 MAJOR로 표시합니다. 0.x 개발 버전에서는 호환성 변화도 릴리스 노트에 명시합니다.

### 0.7.0 ComfyUI bridge and runtime model names

- Add an optional CPU-only ComfyUI Qwen 2.1 Turbo INT8 bridge for existing port 8765; preserve original crop dimensions and exact outside-mask pixels.
- Keep the user's installed ComfyUI/SageAttention and external vLLM unchanged. Explicitly preload configured Comfy filenames; never globally interrupt or unload another Comfy job.
- Add editable diffusion/encoder/VAE, OCR and detector model names. Compatible local Vision bundles can change via bind-mount configuration and restart, without per-model app or image builds.

### 0.8.0 ComfyUI inference extension candidate

- Host Qwen Turbo, RF-DETR and Hayai API routes inside the existing ComfyUI server; inference and warmup run through its normal queue.
- Add one optional ComfyUI origin to Koharu Custom settings. Keep local cropping, flat bubble filling and final page compositing.
- Configure model filenames and verified Vision directories on the server, with explicit warmup and resident Vision reuse.
- Keep original crop dimensions and exact source pixels outside the compositing mask; own queued jobs across client timeout without global interruption.
- Include the extension and installation instructions in source packages. GPU/host installation remains unverified.

### 0.9.0 Web UI workflow ownership

- Add a ComfyUI Web UI panel to open, edit, atomically apply and warm up active detection/OCR/inpaint workflows. Saved graphs apply to the next submission without client rebuild or server restart.
- Separate Vision image input, detector/OCR inference and output contracts, permitting intermediate IMAGE/STRING preprocessing and postprocessing nodes.
- Move all inference controls into workflow widgets. Simplify Koharu Custom to one ComfyUI origin and save/check actions.
- Persist active UI/API snapshots outside tracked templates; retain original mask/coordinate boundaries and prompt-specific queue ownership.
