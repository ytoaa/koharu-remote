# Koharu ComfyUI: server-only image-operation refactor

This update works with the installed 0.9.0 Koharu client. It changes only ComfyUI Python nodes, workflows and the browser extension. It keeps the RGB PNG HTTP response and requires no Windows build, container rebuild or new client header. 이전 클라이언트 계약 실험 was closed without merging at the user's request; its client coverage-contract proposal is discarded.

## Update

```bash
cd ~/koharu-remote
git switch main
git pull --ff-only
```

Keep existing config.json, profiles, model files, symlink, launch flags and venv. Restart ComfyUI once and hard-refresh its browser page. Export/back up your existing workflow first. In the Koharu sidebar select the stage and click **새 기본 템플릿 열기**. Restore your prompt/sampler/custom connections, then click **현재 그래프를 Koharu에 적용**. The template retains current configured Qwen model filenames. Opening a template never overwrites a saved profile; applying it does. Existing profiles are not silently migrated.

Repeat for detection/OCR if using them: KoharuInput's uploader/max_side/use_client_settings widgets were removed. The inpaint output's mask_dilation widget was removed. Existing saved graphs need their new input/output nodes and explicit standard operations, or the new template. Use the model-preparation button after applying the graph. Already submitted requests retain their snapshot.

## Graph ownership

| Operation | Node |
|---|---|
| Manual image upload | Standard LoadImage |
| API image/mask input | KoharuInput (unmodified image/mask + request ID) |
| Image resizing | Standard ImageScale (width/height 0 means unchanged) |
| Mask dilation/erosion | Standard GrowMask |
| Limit to installed client's original mask | Standard MaskComposite, multiply |
| Source/generated blending | Standard ImageCompositeMasked |
| Image/mask preview | Standard PreviewImage, MaskToImage |
| Final HTTP response | KoharuInpaintOutput (RGB PNG encoding only) |

Qwen model loading, encoding, sampling, sigmas, CFG, seed and VAE decoding remain ordinary workflow nodes. KoharuInput no longer resizes or rounds dimensions. For manual GUI testing connect standard LoadImage.image/mask to KoharuInput and leave request_id empty. During API execution these manual loader connections are removed so test uploads are not required. Core LoadImage's absent 64x64 zero mask is represented at source dimensions; real masks are never implicitly resized.

The default graph composites decoded output onto the original-sized source with ImageCompositeMasked.resize_source=true. Connect its output to both PreviewImage and KoharuInpaintOutput. The output boundary does not resize, dilate, blend or write preview files. Final dimensions must match request dimensions. KoharuInput.mask → GrowMask → MaskComposite (multiply with KoharuInput.mask) → ImageCompositeMasked.mask makes the current client limit explicit. Inspect the effective mask using MaskToImage → PreviewImage.

**Installed-client limit:** Koharu still performs its native flat-region filling, tile/component restriction and final original-mask application. Expanded masks and filled holes outside that original mask cannot take effect on the final page with this server-only update. The default MaskComposite intersection avoids presenting an expanded result that the client will discard. Native tiling can further restrict context coverage; overlapping crops do not make server-only mask expansion authoritative. Mask expansion made only during inpainting remains limited by the installed client. Detection-stage corrections, described below, instead become the mask received by that client. This update exposes operations and preserves the existing protocol; it does not claim to solve original-mask omissions end to end.

KoharuMaskFillHoles is available as an independent optional operation because core mask nodes do not provide enclosed-hole filling. It has an enabled toggle and a pixel-area cap, preserves exterior-connected regions and larger holes, and is not connected in the default installed-client graph. Use it for manual ComfyUI experiments; filled holes remain subject to the installed client limit. Core FeatherMask feathers the rectangular mask canvas edges, not arbitrary lettering contours.

## Detection and OCR

KoharuDetector/KoharuOCR own model inference and model-specific options that core ComfyUI nodes do not provide. Dedicated loader nodes now own bundle selection/loading and supply typed model inputs to those execution nodes. Detection output must use original page coordinates; the detector no longer silently resizes preprocessing back to source size. Crop/rotation/resizing changes need explicit coordinate restoration. OCR text can pass through STRING processing nodes before KoharuOCROutput; retain metadata to preserve model identity. Arbitrary replacement architectures require suitable installed nodes.

The existing verified adapters are reused in the same ComfyUI process. No vLLM/torch/CUDA changes are made. The enclosed-hole node uses scipy, already a dependency of ComfyUI's core GrowMask. Model preparation remains in the normal ComfyUI queue.

## Workflow editing and live inspection

The Koharu sidebar opens/applies stage graphs, prepares models and enables live following. Independent inpaint/detection/OCR checkboxes choose which stage takes over the canvas. Selected browser sessions receive native execution/progress/PreviewImage events. Repeated requests with the same saved graph reuse the canvas. Existing workflow tabs/files are not deleted.

PreviewImage writes ComfyUI temporary preview files intentionally, so live and completed-job viewing can work. Output nodes do not create additional previews. Request images/results otherwise remain in memory and are released after owned execution completes or is confirmed removed. Up to four requests are owned concurrently. A timeout does not globally interrupt unrelated GUI work.

Applied profiles live in koharu_nodes/profiles (or KOHARU_WORKFLOW_DIR), separate from tracked templates. Each request reads its own saved graph snapshot. Unapplied canvas edits do not affect production. Config.json/model files/profiles are not included in source packages. An optional KOHARU_COMFY_TOKEN protects extension APIs; the browser field is not persisted in graphs.

`--preview-method none` disables sampler latent previews, not PreviewImage results or progress events. Keep your current launch flags. Historical jobs cannot gain preview references retroactively, and released API inputs cannot be replayed without supplying images for a new manual test.

## Validation

31 focused offline extension tests passed, including unmodified GUI input, standard graph binding, RGB output without hidden compositing, dimension rejection, mask-hole filling, request ownership and an installed-client request with no new header. Python compilation and browser-extension syntax checks passed. Live GPU/page behavior of this refactor has not been verified locally.

## Hayai OCR repetition recovery

KoharuOCR exposes optional `retry_on_repetition` (default true) and `retry_repetition_penalty` (default 1.1). Initial generation retains the existing repetition_penalty (default 1.0) and other workflow controls. Only an excessive-repetition validation failure is retried, at most once, and only when the configured retry penalty is greater than the initial penalty. Token-limit/other failures are not retried. The retry preserves token budget, beams, length penalty and early-stopping settings. Recovery attempts are logged without recognized text; successful result metadata includes the recovery penalty and attempt count.

Validation stays strict: rejected text is not trimmed, substituted with empty text or passed into translation. Persistent failures return HTTP 422 with structured codes `ocr_repetition_loop`, `ocr_token_limit` or `ocr_invalid_output`, instead of an unclassified 500. The installed client still displays its generic 422 message; the precise code is available in server responses/logs. Arbitrary ComfyUI/model exceptions remain errors and are not reclassified as repetition failures.

This is a server-only change. Pull the sidebar branch, restart ComfyUI and prepare models again. Older saved KoharuOCR graphs without the optional controls use the above defaults; recreate the OCR node to expose controls if the browser retains its older widgets. Model-specific repetition recovery quality requires live crop testing; offline tests validate bounded attempts, strict rejection and error transport.


## Detection-mask extraction and correction (no client rebuild)

The default detection template now uses:

`KoharuDetector → KoharuDetectionMasks → KoharuMaskFillHoles → GrowMask → KoharuDetectionMasksApply → KoharuDetectionOutput`

KoharuDetectionMasks extracts a list of local instance MASK crops plus their coordinate context. Select text/effect labels independently; the default is **text + onomatopoeia**, leaving bubble/panel masks unchanged. Default padding is 8 pixels, clipped at page edges. Increase padding before using growth larger than that margin; processing never implicitly resizes or reorders crops. Local masks avoid allocating one full-page float image per detection. Total extracted area is bounded at 16 Mi pixels and final RLE at four million runs. Empty selections use a no-op 1x1 mask so core nodes with scalar widgets can run during warmup.

KoharuMaskFillHoles defaults enabled with a 4096-pixel enclosed-hole cap. Core GrowMask defaults to expand=0: start by testing filling alone. KoharuDetectionMasksApply rejoins the MASK list with its context, thresholds at 0.5 and returns the same protocol-1 detection result with selected masks replaced. Changed extents are encoded in original page coordinates; expand_bbox=true expands boxes as needed to cover expanded mask pixels. Label, score, result metadata and unselected masks stay intact. Original detector results are not mutated. A changed instance count or crop size is rejected rather than guessed. Use MaskToImage → PreviewImage to inspect the processed local masks.

This correction happens **before** the installed Koharu client builds its page text mask. Native detection uses the returned segmentation masks to construct that mask (including its existing morphological closing), so filled pixels are already present in the client's initial mask instead of being late additions discarded by inpaint compositing. Region selection can still clip them; an explicit manual/transient inpaint mask overrides detection-derived masks. The change does not solve generated/source brightness mismatch.

To activate: pull main, restart ComfyUI, refresh the browser, select **감지 → 새 기본 템플릿 열기**, select the detector loader bundle and correction controls, apply the graph and prepare detection. Existing saved graphs are preserved; alternatively add/wire the two new mask bridge nodes and standard processing nodes yourself. **Re-detect existing pages** before testing cleanup: previously stored masks are not rewritten automatically. No Windows build is needed.

Focused tests cover hole fill + growth RLE roundtrip at original coordinates, preserved untouched classes/original results, empty detections, malformed RLE, crop/count mismatch and memory bounds. Native mask consumption was inspected in pinned upstream source. The updated graph still needs live ComfyUI/GPU and final-page verification; offline tests do not establish image quality.


## OCR input patch budget

KoharuOCR now exposes optional max_patches=256/512/1024, default 256 for existing workflow behavior. The pinned Python Hayai decoder derives visual-prefix length from input features, and SigLIP2 NaFlex interpolates positions using spatial shapes; the server can therefore pass a larger patch budget without replacing weights or modifying release model code. This changes preprocessing resolution and can change recognition/runtime/memory; it is opt-in and has not been validated on the user's GPU.

For the reported 1084x403 paragraph, budgets 256/512/1024 produce 400x160, 576x224 and 832x304 model inputs respectively. Start with 1024 in the OCR node for this small-letter paragraph, use max_new_tokens=512 if needed for text length, then apply the workflow. max_new_tokens alone does not affect image resolution. Model logs show crop size, resized model input and budget. Optional retry uses the same selected budget. If the browser retains old widgets, recreate KoharuOCR or open the new OCR template and restore your settings. Older saved profiles remain at 256 until edited. No client rebuild is needed.

If this still loops, test line crops next; increasing resolution does not prove a paragraph is supported or that the repetition guard is correct for every crop. Strict output validation remains enabled. Offline checks cover spatial shape, patch padding, supported budgets and retry wiring; model quality at the expanded budgets remains a live validation item.


## Paragraph OCR line crops

For inputs still failing at 1024 patches / 512 text tokens, the new OCR template exposes:

`KoharuInput → KoharuOCRLineCrops → KoharuOCR → KoharuOCRJoin → KoharuOCROutput`

LineCrops returns an IMAGE list, which ComfyUI maps through KoharuOCR individually. OCRJoin receives the STRING/metadata lists and restores their reading order. No client change is needed. The template includes PreviewImage on the crops and starts with 512 patches per line, 256 output tokens per line. You can retain your current 1024/512 settings instead when wiring the two new nodes into an existing graph. Each line may perform the existing single repetition retry; total time depends on line count. A failed line fails the request rather than silently returning partial text.

Splitting is optional (`enabled`). `direction=auto` chooses horizontal for a clearly wide crop and Japanese right-to-left columns for a clearly tall crop; ambiguous aspect ratios pass through. Explicit horizontal/vertical_ltr/vertical_rtl choices are available. The splitter uses local background contrast and ink projections, not a learned layout model. `polarity`, `background_window`, `contrast`, `ink_fraction`, `min_gap` and `min_line_extent` control band detection. Small preview fonts may need a smaller min_gap/min_line_extent. Inspect the actual crop previews before relying on it. Multiple-column layouts, drawings and mixed orientations require careful region selection or disabling automatic splitting.

Source coverage is retained: cuts occur in detected gaps, all pixels belong to at least one crop, and padding adds context. Keep padding small relative to min_gap to avoid duplicating nearby lettering. If fewer than two or more than max_lines bands are found, the source passes through intact. Default max_lines=32 bounds the number of crops. Join checks crop/result counts, model identity and text metadata, then joins with newlines within the existing result-size limit.

After pulling/restarting, choose **OCR → 새 기본 템플릿 열기**, select the OCR loader bundle, inspect/edit controls, apply and prepare OCR. This opens a new graph without overwriting existing profiles until applied. Logs report segmentation direction/crop count and each actual Hayai input resolution. Older saved graphs continue their old single-crop path until rewired/applied.

44 focused extension tests passed, including horizontal/vertical ordering, retained pixel coverage, disabled/blank/ambiguous/excess-band pass-through and rejected missing/mismatched line results. Splitting was also inspected on a text area from the supplied screenshot. These are geometry/contract checks, not validation of live OCR quality on the original failing crop.


## Dedicated OCR/detection model loaders

KoharuDetectorModelLoader → KoharuDetector, and KoharuOCRModelLoader → KoharuOCR. Execution nodes no longer contain model_directory widgets or initialize models. Loader dropdowns list named compatible **model bundles**, not individual weight files, through ComfyUI's folder-path registry. Qwen/inpaint loaders and paths are unchanged.

| Stage | Default bundle folder | Registered category |
|---|---|---|
| Detection | ComfyUI/models/koharu/detection/<bundle>/ | koharu_detection |
| OCR | ComfyUI/models/koharu/ocr/<bundle>/ | koharu_ocr |

Detection inventory requires `model.safetensors`, `load_model.py` and `inference_config.json`; OCR inventory requires `model.safetensors`, `modeling_hayai.py` and `config.json`. Detection bundles do not need an OCR-style `config.json`.

Bundles retain their existing model.safetensors, configuration, verified adapter code and tokenizer/other required files. Complete model folders can be placed here directly or registered by a symlink. Existing model contents are not moved, replaced or downloaded by the extension. For the verified DGX installation:

```bash
mkdir -p /opt/ComfyUI/models/koharu/detection
mkdir -p /opt/ComfyUI/models/koharu/ocr
ln -s /srv/koharu/models/koharu-layout-rfdetr-seg-2xl-1152 /opt/ComfyUI/models/koharu/detection/koharu-layout-rfdetr-seg-2xl-1152
ln -s /srv/koharu/models/hayai-ocr-v2 /opt/ComfyUI/models/koharu/ocr/hayai-ocr-v2
```

If a destination already exists, inspect it; these commands do not force-overwrite anything. Alternatively register parent bundle folders via normal ComfyUI extra_model_paths.yaml:

```yaml
koharu_models:
  base_path: /srv/koharu/models
  koharu_detection: .
  koharu_ocr: .
```

Inventory distinguishes detection/OCR by their adapter files and supports nested folders, symlink bundles and registered external roots. Inventory does not load weights or perform GPU inference. Loaders retain the existing verification/warmup and resident adapter cache; selecting the same resolved bundle reuses it, and switching selections retires the previous runtime-cache entry for that stage. Readiness tracks the currently connected loader selection without loading a model in the health route. Compatible checkpoints still need the verified bundle format; this does not make arbitrary architectures interchangeable.

Pull/restart/refresh after registering folders. In **감지** and **OCR**, open **새 기본 템플릿 열기**, select the loader bundle, restore your preferred mask/line/OCR settings, apply and prepare models. Existing profiles are preserved and need rewiring/reapplying: replace the old path field with a typed loader connection. The obsolete combined KoharuVision node is removed. Old detector_path/ocr_path config keys may remain in an existing config but no longer select models. No Windows/client rebuild is required.

46 focused extension tests passed: registered bundle inventories, nested/linked/external paths, cyclic-link traversal, resident reuse/switch, typed inference connections and selected-model readiness. Full model loading in the changed graph still needs live DGX verification.

## Local background color matching

GUI workflow: `comfyui/workflows/qwen21-turbo-local-color.json`. Pull/restart ComfyUI, load this file and apply it to the inpainting stage through the Koharu menu. Existing saved profiles are not overwritten. No client rebuild, model download or new Python dependency is required. The updated template omits the artificial Sleep delay and its KJNodes dependency; Acly's color-match extension is not used in this workflow.

Core ImageCompositeMasked first aligns the decoded image to source dimensions (no mask, resize_source=true). KoharuLocalColorMatch then estimates smooth additive offsets using unmasked reference/generated pixels in similar-luminance bands, keeping white margins from dominating gray panels. Masked samples and pairs differing beyond sample_tolerance are excluded. Pixels without nearby samples inherit a supported offset of similar luminance; without sufficient reference samples, output is unchanged. Calculations use a bounded sampling grid and introduce no resident cache.

Controls: strength (0 disables, default 1), radius (local smoothing range in source pixels, default 32), max_shift (maximum RGB offset, default 0.10), sample_tolerance (maximum reference/generated difference used as a sample, default 0.15), mode (luminance or rgb; default luminance). Exclusion-mask hole filling and expansion affect sampling only. Core final compositing uses the original input mask and resize_source=false after explicit alignment. Detection-mask omissions must still be corrected in the detection graph; this node cannot reconstruct background patterns or panels discarded by the generator.

Five labeled previews show source, original mask, aligned VAE output, locally corrected output and final composite. Compare strength 0/1 with the same seed. Numerical fixtures verify offsets, mixed backgrounds, excluded lettering, correction bounds, no-op cases, sampling and graph-node dimensions. Actual DGX image quality remains to be evaluated.

OCR failures distinguish repetition before checking the token cap, so a repeated output that consumes the full budget can use the existing single repetition-penalty retry. Ordinary truncated text remains `ocr_token_limit` and is rejected without an automatic budget increase. Failure diagnostics log crop dimensions, decoded-token count/configured limit, character count, patch budget, repetition penalty and beam count, without recognized text. The limits and line-segmentation controls in the applied OCR graph remain authoritative.

OCR line `direction=auto` analyzes row/column projections for near-square crops instead of passing them through solely because their aspect ratio is below 1.25. It chooses a uniquely plausible direction or one with clearly greater inter-band separation; ambiguous grids remain intact. Wide/tall crop defaults and explicit directions are unchanged. Pass-through diagnostics include reasons (disabled, low_contrast, insufficient_bands, no_separated_rows_or_columns, ambiguous_rows_and_columns). For known horizontal paragraphs, explicitly choosing horizontal bypasses direction ambiguity; the output token cap remains an independent workflow setting.

## SAM-TS-L candidate text masks

Candidate GUI graph: `comfyui/workflows/detection-sam-ts-l.json` (matching API graph is beside it). Existing detection/inpainting/OCR profiles remain unchanged until the user applies this graph to **감지**. RF-DETR still supplies region labels, boxes, bubbles and panels; SAM-TS-L supplies text pixels. The existing protocol-1 API and client build remain unchanged.

Prepare in the existing ComfyUI venv:

```bash
source /opt/ComfyUI/.venv/bin/activate
cd ~/koharu-remote
git switch main
git pull --ff-only
uv pip install --no-deps einops
python3 comfyui/prepare_text_sam.py --models-dir /opt/ComfyUI/models --download
```

Only the full `mayocream/koharu-text-sam-ts-l` SafeTensors weight and the pinned config/inference example are downloaded (about 1.36 GB of weights). The adapter-only file and original SAM/TextSeg `.pth` weights are not required. The preparation script preserves existing files and rejects mismatched or incomplete downloads. It verifies SHA256 and does not install packages, start containers, touch vLLM, or load CUDA. `einops` is the only additional inference dependency beyond the current Torch/numpy/Pillow/safetensors stack; `--no-deps` prevents this package command from changing Torch/CUDA. Do not install Hi-SAM's historical training requirements into ComfyUI.

Restart ComfyUI and refresh the browser. Open the candidate graph, select **감지** in the Koharu menu, apply the graph and prepare detection. A new detection pass is required to update already-detected pages. The registered model category is `koharu_text_masks`, default path `ComfyUI/models/koharu/text_masks/<bundle>/`. Extra model paths and symlinks work as with the other loaders. Detection health also checks the selected SAM bundle and precision, without loading it in an HTTP handler.

Nodes:

- **KoharuTextMaskModelLoader**: selects the resident model and `bf16` mixed precision (release default) or `fp32`. BF16 retains FP32 weights and autocasts inference. If CUDA BF16 operations fail on a particular runtime, select fp32 and prepare again.
- **KoharuTextMask**: `full_page` matches published RGB 0–255 preprocessing, longest side 1024, bottom/right gray padding 128, HR mask restoration to original coordinates. `threshold=0` matches the release threshold; lower values include more pixels. `tiles` uses overlapping 512/768/1024-pixel source crops and weighted logit joins, with a 64-tile and 16-Mpixel page limit. Tiles can improve pixel resolution for small text at an additional inference cost; actual gains/speed are not yet measured on DGX. Progress is sent through ComfyUI's ordinary progress events.
- **KoharuTextMaskApply**: `enabled=false` returns original RF-DETR results for comparison; SAM previews still execute. `replace` replaces selected text/SFX masks inside padded region bounds; `union` preserves original pixels and adds SAM pixels. `empty_fallback=true` retains an existing mask when SAM finds none in that region. Nonselected labels and original inputs are preserved.
- **recover_missed**: off by default. When enabled, unclaimed SAM pixels outside selected region bounds are grouped into additional text instances, controlled by grouping_radius/min_component_area/max_new_regions. Their score is 0 with `score_kind=unscored_mask_component`, because SAM-TS has no calibrated instance confidence. The existing client's 160-instance limit is enforced. Wrong text-like art or poor paragraph grouping are possible; this is an optional recovery mechanism, not a semantic text detector.
- **KoharuDetectionMaskPage**: RLE-to-page mask bridge for previews/standard image operations. It does not perform inference or morphology.

Final cropped masks still pass through existing hole filling, core GrowMask and the original-coordinate reinsertion node. Core EmptyImage/ImageCompositeMasked/ImageBlend create a green overlay for review. Previews show RF-DETR mask, full SAM mask, final client mask and overlay. Raw SAM pixels outside selected boxes do not reach the client unless recover_missed is enabled. Start with full_page/replace and compare enabled false/true on the same page; use text-only labels if SAM weakens stylized SFX.

Model release is pinned to `5dd97423e0fbf2404264979136d47e8101144046`; model/config/example hashes are verified. Minimal Hi-SAM inference modules are vendored from `ymy-k/Hi-SAM@69009434d4dba5541f228d8f5acb0754c333d417` with Apache-2.0 LICENSE/NOTICE. Only inference activation-checkpoint calls are bypassed locally; no global Torch patch, pickle loading or runtime network downloads are used. Meta-device architecture verification matches 728 state tensors / 338,934,897 elements. Numerical/contract tests pass; actual full-weight loading, GPU execution and quality still require live DGX validation.

## Diagnose a line present in detection previews but absent from inpaint input

The installed client's detection stage runs a second same-label box NMS (IoU >= 0.5), then uses **text** instances to build its automatic text mask, with page-dependent dilation and closing. The ordinary KoharuDetectionMaskPage preview unions masks **before** that NMS and morphology. It is not a preview of the client's completed automatic mask. Its labels selector affects this visualization only; it does not filter or relabel the HTTP detection response.

KoharuDetectionMaskPage now has an optional `client_nms` switch, default false. Set labels=text and compare false/true on the same detection result. True projects the installed client's NMS, including float32 boxes/scores and its stable score/top/rightmost/label ordering; it still does not emulate dilation, closing, selected-region clipping, flat filling, or tile splitting. Recreate this preview bridge node after restarting/refreshing if the optional widget is missing. The switch never modifies the response or existing saved graphs. No native client build is required.

Every KoharuDetectionOutput logs a read-only `Koharu client mask audit` for manual and API execution. It lists suppressed indices, owning indices, scores, boxes, IoU, and the union's unique text pixels/bounds lost by NMS. Duplicate boxes with identical mask coverage report zero lost pixels. Pixel comparison is bounded to 16-Mpixel pages; larger pages retain the box report with pixel audit explicitly skipped. No image, mask, or diagnostic file is retained. A diagnostic failure does not change the detection response. A nonzero loss identifies a concrete potential client-boundary discrepancy; it does not alone prove the user's particular missing line has that cause.

`Koharu graph` logs the actual stage/profile, a stable API-graph SHA256 prefix, request ID and reachable node classes. `Koharu detector input` logs manual/API ownership, source dimensions and an RGB SHA256 prefix. These distinguish a manual graph from the applied server profile or a differently cropped/resized source, without logging image content or prompts. When a new project still reproduces the problem, run its detection once and inspect these logs before changing model thresholds or compositing.

## Preserve masks of overlapping text detections

A live 2150x3035 page reported 22 detections, 13 survivors after the installed client's NMS, and 2142 unique text pixels lost at [1471,2697,1634,2721]. This explains the footer line being visible in the server's union preview but absent from the inpaint input. The diagnostic applies the actual client's same-label box test; it does not infer the cause from model quality or prompt wording.

`KoharuDetectionMergeOverlaps` resolves this boundary on the server. Place it **after all instance mask refinements**, before KoharuDetectionOutput, and feed the final page previews from its output too. For each same-label NMS suppression at IoU >= 0.5 it retains the winning instance's label/confidence/metadata and unions the suppressed instance's mask into it. Default labels=text leaves effects, bubbles and panels unchanged; select text + onomatopoeia to preserve effect masks too. It neither relabels regions nor creates extra foreground pixels. enabled=false returns the original result for comparison.

expand_bbox=true expands the retained OCR box to cover the joined pixels. Such expansion can create additional overlaps, so merging repeats until no selected instance can be suppressed by the client's NMS. Each pass reduces the number of instances, bounded by the client's 160-instance limit. Local allocations are bounded at 16 Mi pixels and final RLE at four million runs. expand_bbox=false preserves original retained boxes and is available for comparison, but OCR crops can then omit the added pixels. Merged paragraphs can change OCR grouping; the mask union itself is preserved exactly.

Both the fresh default detection template and `comfyui/workflows/detection-sam-ts-l.json` now contain this node before their terminal result. The candidate's ③/④ show text-only masks from the merged output. Existing applied profiles are deliberately not modified by pulling code: add the node to your graph, or open the updated candidate, restore your current model/settings/connections, then **apply to detection**. Restart ComfyUI to register the new node, refresh the browser, and re-detect a fresh page. Existing project masks will not be rewritten by changing an inpaint graph. No Windows build or inpaint workflow change is required.

Tests reproduce an overlapping low-score footer, verify identical mask unions before/after, preserve winner confidence/unselected classes/input objects, and cover box-expansion cascades and original-coordinate limits. The user confirmed the missing line was included after applying this correction on DGX.

## 최신 기본 템플릿과 처리 대상 선택

서버를 업데이트하고 ComfyUI를 재시작한 뒤 브라우저를 새로고침하세요. Koharu 메뉴에서 감지 또는 인페인트를 선택하고 **새 기본 템플릿 열기 → 현재 그래프 적용**을 누르면 최신 구성이 반영됩니다. 현재 적용 중인 프로필은 업데이트만으로 덮어쓰지 않습니다. 클라이언트 빌드는 필요 없습니다.

- **감지**: 제공했던 SAM-TS-L 워크플로우를 기본으로 사용합니다. RF-DETR 영역 감지 → SAM 글자 마스크 → 기본 GrowMask 등 마스크 보정 → 처리 대상 선택 → 겹친 영역의 마스크 병합 → 최종 출력 순서입니다. SAM 모델은 `models/koharu/text_masks/`에 설치되어 있어야 합니다. 모델 목록에서 기본 이름을 찾지 못하면 설치된 첫 모델을 템플릿에 표시하므로, 적용 전에 로더 선택을 확인하세요.
- **인페인트**: Qwen-Image-2.1-Turbo 8-step, 원본 크기 정렬, 국소 밝기 보정, 기본 ImageCompositeMasked의 최종 합성을 포함합니다. 보정용 제외 마스크의 채움·확장은 색 보정 참조에만 사용합니다. 최종 합성은 입력 마스크를 그대로 사용합니다. 프롬프트·seed·색 보정 강도·참조 범위·최대 보정량은 그래프에서 조절합니다. 인위적인 Sleep 대기는 제거했습니다.
- **OCR**: 기존 최신 줄 분리 → Hayai → 읽기 순서 결합 구성을 유지합니다. 줄 분리 방향, 패치 수, 생성 토큰 수, 반복 억제와 재시도 옵션을 그래프에서 조절합니다. 토큰 제한은 잘못된 결과를 허용하는 옵션이 아닙니다.

**Koharu 처리 대상 선택**은 마스크 보정 뒤, 겹친 영역 병합 앞에 놓습니다. 본문·말풍선·패널을 각각 포함할 수 있고, 효과음은 다음 세 가지로 선택합니다.

| 효과음 선택 | 기존 클라이언트에서의 동작 |
| --- | --- |
| 기존 분류 유지 (`keep`) | `onomatopoeia` 분류를 유지합니다. 자동 본문 OCR·인페인트 대상에 추가하지 않습니다. |
| 본문처럼 처리 (`as_text`) | 마스크와 좌표·점수는 유지하고 `text`로 전달합니다. **OCR·번역·인페인트 모두**에 포함됩니다. 원래 분류는 진단용 메타데이터에 보존합니다. |
| 제외 (`exclude`) | 감지 응답에서 효과음을 제외합니다. |

기존 클라이언트는 같은 `text` 분류로 OCR과 자동 정리를 수행하므로, 서버 옵션만으로 효과음 **지우기만**을 별도 제공하지 않습니다. 본문 포함을 꺼도 `as_text`로 선택한 효과음은 처리됩니다. 낮은 확률의 오탐이나 장식 그림을 효과음으로 판단한 경우도 포함될 수 있으므로 최종 마스크 미리보기를 확인하세요. 라벨 변경 뒤에 병합하므로 본문과 겹친 효과음도 클라이언트 중복 제거에서 고유 마스크 픽셀을 잃지 않습니다.

Koharu 커스텀 노드의 이름·설명·입출력·선택지는 ComfyUI 기본 번역 형식 `locales/ko/nodeDefs.json`으로 한국어를 제공합니다. 입력 이름과 선택지는 ComfyUI 언어를 한국어로 설정하면 표시됩니다. OCR, SAM, IoU, bf16/fp32, 모델 파일명, 모델용 영어 프롬프트 등 더 명확한 표기는 유지합니다. 실행용 노드 ID·입력 키·저장된 enum 값은 바꾸지 않아 기존 프로필과 API 연결을 유지합니다. 다른 확장의 노드 번역은 각 확장에 따릅니다.
