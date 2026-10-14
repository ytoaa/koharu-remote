# Third-party sources and notices

## Koharu

원본: https://github.com/koharu-rs/koharu

고정 기준: 버전 0.83.5, commit `52bb4f55e2266f72c1523562a583c827ec191ceb`. `upstream.json`과 `tools/checkout.py`가 같은 revision을 사용합니다. 이 저장소는 패치, 별도 Remote 클라이언트, 서버 어댑터와 ComfyUI 워크플로우를 제공합니다.

원본 라이선스: MIT OR Apache-2.0. `LICENSE-MIT`와 `LICENSE-APACHE`의 원문을 보존합니다. MIT 저작권자는 Mayo Takanashi and Koharu contributors입니다. 별도 빌드는 공식 Koharu 배포물이 아닙니다.

## Hi-SAM inference modules

원본: https://github.com/ymy-k/Hi-SAM

고정 revision: `69009434d4dba5541f228d8f5acb0754c333d417`.

포함 위치: `dgx/vision/_text_sam/`. Apache-2.0 원문, 원본 파일의 저작권 헤더, `NOTICE`를 보존합니다. 로컬 수정은 추론용 image-encoder의 activation-checkpoint 경로에 국한되며 기존 NOTICE에 설명되어 있습니다. 학습 코드·데이터셋·pickle checkpoint는 배포하지 않습니다.

## RORem service adapter

가져온 서비스 원본: https://github.com/ytoaa/koharu-companion-runtime-adapter

고정 revision: `ced08d90bc06dece6792b1cf3f61b623615f2ae1`, 서비스 0.2.3. 포함 위치: `dgx/rorem/`. 파일별 원본·가져온 버전·현재 SHA256과 변경 revision은 `dgx/rorem/provenance.json`에 있습니다. 해당 서비스 원본은 비공개 저장소이므로 링크 열람에 권한이 필요할 수 있습니다. 외부 모델의 사용 권한은 이 서비스 코드의 출처 표기와 별개입니다.

## External model bundles

가중치·학습 데이터는 이 Git 저장소와 소스 ZIP에 포함하지 않습니다. 다운로드 도구는 고정 revision의 외부 파일을 받아 checksum을 검사하며 사용자는 해당 배포자의 조건에 따라 파일을 사용합니다. 아래의 라이선스 표기는 2026-10-14에 배포자 모델 카드에서 확인한 식별자입니다. 설치·재배포 시 사용하는 고정 revision의 라이선스 원문을 확인하세요.

| 용도 | 배포자/모델 | 고정 revision | 배포자가 표시한 라이선스 |
| --- | --- | --- | --- |
| 레이아웃 감지 | [mayocream/koharu-layout-rfdetr-seg-2xl-1152](https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152) | `aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2` | `other` (모델 카드의 별도 조건) |
| OCR | [JustANormalTinkerer/hayai-ocr-v2](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2) | `4a4ce477c9a8841f208b94e1d9ed5c0938965e05` | Apache-2.0 |
| OCR vision config | [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) | `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` | 원본 모델 카드/배포 조건 참고 |
| 글자 마스크 | [mayocream/koharu-text-sam-ts-l](https://huggingface.co/mayocream/koharu-text-sam-ts-l) | `5dd97423e0fbf2404264979136d47e8101144046` | Apache-2.0 |
| 선택형 인페인트 | [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | `fc13ba1986fcb905854dc8de2f52eee395f49b5b` | Qwen Research License Agreement |

RF-DETR 소프트웨어의 Apache-2.0 표기는 학습 이미지의 권리를 부여하지 않습니다. 감지 모델 카드는 Manga109의 학술 이용 조건을 별도로 명시합니다. SAM-TS-L 모델 카드도 학습 마스크의 CC BY 4.0 attribution과 Manga109 이미지의 별도 조건을 설명합니다. 원본 모델 카드에 있는 attribution과 사용 조건을 함께 확인하세요.

ComfyUI의 양자화 Qwen 모델, text encoder, VAE와 기존 RORem pipeline/LCM LoRA는 사용자가 외부에서 준비하는 파일입니다. 이 저장소의 예제 파일명은 해당 가중치의 재배포 허가를 부여하지 않습니다. 모델을 배포물에 추가한다면 해당 배포자·원본 모델·변환본의 조건을 별도로 확인해야 합니다.

## Runtime dependencies

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI): 외부 실행 환경. custom nodes와 워크플로우가 API로 연결됩니다. ComfyUI 자체 소스는 이 소스 배포에 포함하지 않습니다.
- [Diffusers](https://github.com/huggingface/diffusers): 선택형 Qwen backend가 사용하는 revision은 `916a5791ed5f29d966200d2eee53f06862ec4434`입니다. 설치용 요구사항과 모델 manifest에 pin을 보존합니다.
- 나머지 Python/Rust/JavaScript 의존성은 각 패키지의 라이선스가 적용됩니다. Windows 패키징은 upstream이 생성한 runtime resources와 저작권/라이선스 파일을 유지합니다.

개인 경로·주소·커밋 이메일 정리는 제3자의 저작권 고지, 라이선스 원문 또는 원본 프로젝트 출처를 삭제하는 작업이 아닙니다.
