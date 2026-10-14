//! HTTP inference only. Scene ownership and postprocessing stay in Koharu.
use anyhow::{Context, Result, ensure};
use image::{DynamicImage, GenericImageView, GrayImage};
use koharu_ml::koharu_layout_rfdetr_seg_2xl::{
    KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask, KoharuLayoutThresholds,
};
use reqwest::{
    Client, Url,
    multipart::{Form, Part},
};
use serde::Deserialize;
use serde_json::json;
use std::{io::Cursor, time::Duration};

#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Settings {
    enabled: bool,
    detection_enabled: bool,
    ocr_enabled: bool,
    inpainting_enabled: bool,
    comfy_url: String,
    detection_url: String,
    ocr_url: String,
    inpainting_url: String,
    timeout: u64,
    options: Option<ModelOptions>,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: false,
            detection_enabled: true,
            ocr_enabled: true,
            inpainting_enabled: true,
            comfy_url: String::new(),
            detection_url: "http://127.0.0.1:8766".into(),
            ocr_url: "http://127.0.0.1:8766".into(),
            inpainting_url: "http://127.0.0.1:8765".into(),
            timeout: 600,
            options: None,
        }
    }
}
impl Settings {
    fn stage_enabled(&self, stage: &str) -> Result<bool> {
        let selected = match stage {
            "detection" => self.detection_enabled,
            "ocr" => self.ocr_enabled,
            "inpainting" => self.inpainting_enabled,
            _ => anyhow::bail!("Unknown Custom stage"),
        };
        Ok(self.enabled && selected)
    }
}
#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DetectionOptions {
    model: String,
    text: f32, onomatopoeia: f32, bubble: f32, panel: f32,
}
impl Default for DetectionOptions {
    fn default() -> Self { Self { model: "mayocream/koharu-layout-rfdetr-seg-2xl-1152".into(), text: 0.25, onomatopoeia: 0.20, bubble: 0.50, panel: 0.50 } }
}
#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OcrOptions {
    model: String,
    max_new_tokens: u32, repetition_penalty: f32, num_beams: u32,
    length_penalty: f32, early_stopping: bool,
}
impl Default for OcrOptions {
    fn default() -> Self { Self { model: HAYAI_MODEL.into(), max_new_tokens: 256, repetition_penalty: 1.0,
        num_beams: 1, length_penalty: 1.0, early_stopping: true } }
}
#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct InpaintOptions {
    prompt: String, negative_prompt: String, steps: u32, strength: f32,
    guidance_scale: f32, seed: i64, max_side: u32, mask_dilation: u32,
}
impl Default for InpaintOptions {
    fn default() -> Self { Self {
        prompt: koharu_ml::rorem_mixed::DEFAULT_PROMPT.into(),
        negative_prompt: koharu_ml::rorem_mixed::DEFAULT_NEGATIVE_PROMPT.into(),
        // The server runs distilled RORem-4S; do not copy mixed-model steps/CFG.
        steps: 4, strength: 0.99, guidance_scale: 1.0, seed: 0,
        max_side: koharu_ml::rorem_mixed::RoremMixedOptions::default().resolution,
        mask_dilation: 0,
    } }
}
#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct QwenInpaintOptions {
    model: String, text_encoder: String, vae: String,
    prompt: String, seed: i64, max_side: u32, mask_dilation: u32, use_kv_cache: bool,
}
const QWEN_INPAINT_MODEL: &str = "Qwen-Image-2.1-Turbo";
const QWEN_INPAINT_REVISION: &str = "fc13ba1986fcb905854dc8de2f52eee395f49b5b";
fn legacy_inpaint_model() -> String { "RORem-4S".into() }
impl Default for QwenInpaintOptions {
    fn default() -> Self { Self {
        model: QWEN_INPAINT_MODEL.into(), text_encoder: String::new(), vae: String::new(),
        prompt: "Remove all text and sound effect lettering from <image1>. Keep all speech bubbles, their outlines, and white interiors unchanged. Restore the background behind the lettering and preserve the original artwork.".into(),
        seed: 0, max_side: 2048, mask_dilation: 0, use_kv_cache: true,
    } }
}
#[derive(Clone, serde::Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ModelOptions {
    detection: DetectionOptions, ocr: OcrOptions, inpainting: InpaintOptions,
    // Existing 0.4/0.5 options keep their RORem selection and 8765 origin.
    #[serde(default = "legacy_inpaint_model")]
    inpainting_model: String,
    qwen_inpainting: QwenInpaintOptions,
    inpainting_tiled: bool,
}
impl Default for ModelOptions {
    fn default() -> Self { Self { detection: Default::default(), ocr: Default::default(),
        inpainting: Default::default(), inpainting_model: QWEN_INPAINT_MODEL.into(),
        qwen_inpainting: Default::default(), inpainting_tiled: true } }
}
fn official_options() -> Result<ModelOptions> {
    #[allow(unused_mut)]
    let mut options = ModelOptions::default();
    #[cfg(not(test))]
    {
        let config = crate::PipelineConfig::load()?;
        let pipeline = config.read()?;
        let crate::DetectionModel::KoharuLayoutRFDetrSeg2XL(value) = pipeline.detection()?;
        options.detection.text = value.text_threshold.unwrap_or(options.detection.text);
        options.detection.bubble = value.bubble_threshold.unwrap_or(options.detection.bubble);
        options.detection.panel = value.panel_threshold.unwrap_or(options.detection.panel);
        if let crate::InpaintingModel::RoremMixed(value) = pipeline.inpainting()? {
            options.inpainting.prompt = value.prompt;
            options.inpainting.negative_prompt = value.negative_prompt;
        }
    }
    Ok(options)
}
impl ModelOptions {
    fn validate(&self) -> Result<()> {
        ensure!(["RORem-4S", QWEN_INPAINT_MODEL].contains(&self.inpainting_model.as_str()), "인페인트 모델을 확인하세요.");
        for value in [self.detection.text, self.detection.onomatopoeia,
                      self.detection.bubble, self.detection.panel] {
            ensure!(value.is_finite() && (0.0..=1.0).contains(&value), "감지 임계값은 0..1이어야 합니다.");
        }
        for value in [&self.inpainting.prompt, &self.inpainting.negative_prompt, &self.qwen_inpainting.prompt] {
            ensure!(!value.contains('\0') && value.chars().count() <= 2000, "프롬프트 길이 또는 문자를 확인하세요.");
        }
        let o = &self.ocr;
        ensure!((1..=2048).contains(&o.max_new_tokens) && (1..=4).contains(&o.num_beams)
                && o.repetition_penalty.is_finite() && (1.0..=2.0).contains(&o.repetition_penalty)
                && o.length_penalty.is_finite() && (0.1..=2.0).contains(&o.length_penalty),
                "Hayai OCR 고급 옵션을 확인하세요.");
        let v = &self.inpainting;
        ensure!((2..=30).contains(&v.steps) && v.strength.is_finite() && v.strength > 0.0 && v.strength < 1.0
                && (v.steps as f32 * v.strength) >= 1.0, "RORem 스텝·강도를 확인하세요.");
        ensure!(v.guidance_scale.is_finite() && (0.0..=20.0).contains(&v.guidance_scale)
                && (-1..=2147483647).contains(&v.seed) && (256..=1024).contains(&v.max_side) && v.mask_dilation <= 32,
                "RORem 고급 옵션을 확인하세요.");
        for name in [&self.detection.model,&self.ocr.model,&self.qwen_inpainting.model,
                     &self.qwen_inpainting.text_encoder,&self.qwen_inpainting.vae] {
            ensure!(name.len() <= 512 && !name.chars().any(|c| c.is_control()), "모델 이름을 확인하세요.");
        }
        let q = &self.qwen_inpainting;
        ensure!((-1..=2147483647).contains(&q.seed) && (256..=4096).contains(&q.max_side)
                && q.mask_dilation <= 32, "Qwen 고급 옵션을 확인하세요.");
        Ok(())
    }
}
#[derive(Clone, Default)]
struct SessionTokens {
    vision: String,
    inpainting: String,
}
fn secrets() -> &'static std::sync::Mutex<SessionTokens> {
    static TOKENS: std::sync::OnceLock<std::sync::Mutex<SessionTokens>> =
        std::sync::OnceLock::new();
    TOKENS.get_or_init(Default::default)
}
#[cfg(not(test))]
fn config() -> Result<koharu_config::Config<Settings>> {
    static BACKUP: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    BACKUP.get_or_init(|| koharu_config::path().and_then(|p| backup_pre_qwen_config(&p))
        .map_err(|e| e.to_string())).as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    koharu_config::load::<Settings>("custom-models")
}
fn backup_pre_qwen_config(path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    if !path.is_file() { return Ok(()); }
    let backup = path.with_file_name("config.before-qwen-0.6.0.toml");
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
    let mut file = match options.open(&backup) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let written = (|| -> Result<()> {
        file.write_all(&std::fs::read(path)?)?;
        file.sync_all()?;
        Ok(())
    })();
    if written.is_err() { drop(file); let _ = std::fs::remove_file(&backup); }
    written
}
#[cfg(test)]
fn config() -> Result<koharu_config::Config<Settings>> {
    static CONFIG: std::sync::OnceLock<koharu_config::Config<Settings>> =
        std::sync::OnceLock::new();
    Ok(CONFIG
        .get_or_init(|| koharu_config::Config::memory(Settings::default()))
        .clone())
}
fn settings() -> Result<Settings> {
    let mut value = config()?.read()?.clone();
    if value.options.is_none() { value.options = Some(official_options()?); }
    value.options.as_mut().unwrap().inpainting_model = QWEN_INPAINT_MODEL.into();
    Ok(value)
}
pub(crate) fn enabled(stage: &str) -> bool {
    settings().is_ok_and(|s| s.stage_enabled(stage).unwrap_or(false))
}
pub fn get_custom_models() -> Result<String> {
    let value = settings()?;
    let secrets = secrets()
        .lock()
        .map_err(|_| anyhow::anyhow!("Custom credentials unavailable"))?;
    Ok(
        json!({"settings":value,"defaults":official_options()?,"server_defaults":ModelOptions::default(),
        "vision_token_configured":!secrets.vision.is_empty(),
        "inpainting_token_configured":!secrets.inpainting.is_empty()})
        .to_string(),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveInput {
    settings: Settings,
    vision_token: Option<String>,
    inpainting_token: Option<String>,
}
pub fn save_custom_models(input: &str) -> Result<String> {
    ensure!(input.len() <= 32768, "Custom settings too large");
    let mut input: SaveInput = serde_json::from_str(input)?;
    if input.settings.options.is_none() { input.settings.options = Some(official_options()?); }
    input.settings.options.as_mut().unwrap().inpainting_model = QWEN_INPAINT_MODEL.into();
    input.settings.options.as_ref().unwrap().validate()?;
    if !input.settings.comfy_url.trim().is_empty() { origin(&input.settings.comfy_url)?; }
    for (stage, url) in [
        ("detection", &input.settings.detection_url),
        ("ocr", &input.settings.ocr_url),
        ("inpainting", &input.settings.inpainting_url),
    ] {
        if input.settings.stage_enabled(stage)? {
            origin(url)?;
        }
    }
    ensure!(
        (10..=1800).contains(&input.settings.timeout),
        "Timeout must be 10..1800 seconds"
    );
    let mut credentials = secrets()
        .lock()
        .map_err(|_| anyhow::anyhow!("Custom credentials unavailable"))?;
    let mut next = credentials.clone();
    if let Some(raw) = input.vision_token {
        next.vision = token(&raw)?;
    }
    if let Some(raw) = input.inpainting_token {
        next.inpainting = token(&raw)?;
    }
    let config = config()?;
    let previous = config.read()?.clone();
    *config.write()? = input.settings;
    if let Err(error) = config.save() {
        *config.write()? = previous;
        return Err(error);
    }
    *credentials = next;
    drop(credentials);
    get_custom_models()
}
pub async fn check_custom_models() -> Result<String> {
    let mut checked = 0;
    for stage in ["detection", "ocr", "inpainting"] {
        if let Some(remote) = Remote::configured(stage)? {
            remote.ready().await?;
            checked += 1;
        }
    }
    if checked == 0 {
        return Ok("선택한 서버 모델이 없습니다. 현재는 내장 모델을 사용합니다.".into());
    }
    Ok(format!(
        "선택한 서버 모델 {checked}개 연결 확인 완료. 실제 추론 품질은 작업 결과로 확인합니다."
    ))
}

#[derive(Clone)]
pub(crate) struct Remote {
    client: Client,
    url: Url,
    token: String,
    stage: String,
    options: ModelOptions,
}
fn token(value: &str) -> Result<String> {
    let mut value = value.trim();
    if value.is_empty() {
        return Ok(String::new());
    }
    ensure!(value.is_ascii(), "Remote API token must be ASCII");
    while value.len() >= 7 && value[..7].eq_ignore_ascii_case("bearer ") {
        value = value[7..].trim();
    }
    ensure!(
        !value.is_empty() && !value.eq_ignore_ascii_case("bearer"),
        "Remote API token is empty"
    );
    ensure!(
        value.is_ascii() && value.bytes().all(|b| b > 32 && b < 127),
        "Remote API token contains whitespace/control characters"
    );
    Ok(value.into())
}
fn origin(value: &str) -> Result<Url> {
    let url = Url::parse(value.trim()).context("invalid remote origin")?;
    ensure!(
        ["http", "https"].contains(&url.scheme())
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none()
            && ["", "/"].contains(&url.path()),
        "Remote URL must be an HTTP(S) origin"
    );
    Ok(url)
}
fn png(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png)?;
    Ok(bytes.into_inner())
}
fn image_part(image: &DynamicImage) -> Result<Part> {
    Ok(Part::bytes(png(image)?)
        .file_name("image.png")
        .mime_str("image/png")?)
}
impl Remote {
    #[cfg(test)]
    pub(crate) fn test_client(url: &str) -> Result<Self> {
        Ok(Self { client: Client::builder().no_proxy().timeout(Duration::from_secs(10)).build()?,
            url: origin(url)?, token: String::new(), stage: "inpainting".into(),
            options: ModelOptions { inpainting_model: legacy_inpaint_model(), ..Default::default() } })
    }
    pub(crate) fn detection_thresholds(&self) -> KoharuLayoutThresholds {
        let d = &self.options.detection;
        KoharuLayoutThresholds { text: d.text, onomatopoeia: d.onomatopoeia, bubble: d.bubble, panel: d.panel }
    }
    pub(crate) fn inpainting_tiled(&self) -> bool { self.options.inpainting_tiled }
    pub(crate) fn configured(stage: &str) -> Result<Option<Self>> {
        let settings = settings()?;
        if !settings.stage_enabled(stage)? {
            return Ok(None);
        }
        let secrets = secrets()
            .lock()
            .map_err(|_| anyhow::anyhow!("Custom credentials unavailable"))?
            .clone();
        let (url, raw) = match stage {
            "detection" => (&settings.detection_url, &secrets.vision),
            "ocr" => (&settings.ocr_url, &secrets.vision),
            "inpainting" => (&settings.inpainting_url, &secrets.inpainting),
            _ => anyhow::bail!("Unknown Custom stage"),
        };
        Ok(Some(Self {
            client: Client::builder()
                .no_proxy()
                .connect_timeout(Duration::from_secs(5))
                .timeout(Duration::from_secs(settings.timeout))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            url: if settings.comfy_url.trim().is_empty() { origin(url)? } else {
                origin(&settings.comfy_url)?.join(if stage == "inpainting" { "koharu/inpaint/" } else { "koharu/vision/" })?
            },
            token: token(raw)?,
            stage: stage.into(),
            options: settings.options.unwrap_or_default(),
        }))
    }
    fn endpoint(&self, path: &str) -> Result<Url> {
        Ok(self.url.join(path)?)
    }
    fn request(&self, method: reqwest::Method, path: &str) -> Result<reqwest::RequestBuilder> {
        let request = self.client.request(method, self.endpoint(path)?);
        Ok(if self.token.is_empty() {
            request
        } else {
            request.bearer_auth(&self.token)
        })
    }
    async fn status(&self, mut response: reqwest::Response) -> Result<reqwest::Response> {
        let status = response.status();
        if status == reqwest::StatusCode::OK {
            return Ok(response);
        }
        // Only bounded, structured diagnostics; never echo arbitrary server text or credentials.
        let mut body = Vec::new();
        while body.len() < 65536 {
            let Some(chunk) = response.chunk().await.map_err(|e| e.without_url())? else { break };
            body.extend_from_slice(&chunk[..chunk.len().min(65536 - body.len())]);
        }
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
        let detail = &value["detail"];
        let message = match status.as_u16() {
            413 => {
                let limit = detail["max_upload_bytes"].as_u64();
                match limit {
                    Some(bytes) => format!("upload exceeds server limit ({bytes} bytes per file)"),
                    None => "upload exceeds server size limit".into(),
                }
            }
            401 | 403 => "server authentication rejected".into(),
            422 => "invalid image, mask or options; check server logs".into(),
            429 => "server inference is busy".into(),
            _ => {
                let code = detail["code"].as_str().filter(|s| matches!(*s,
                    "cuda_out_of_memory" | "inference_failed" | "model_not_ready"
                        | "image_too_large" | "upload_too_large"));
                code.map(str::to_owned).unwrap_or_else(|| "server request failed; check server logs".into())
            }
        };
        anyhow::bail!("DGX {} HTTP {}: {}", self.stage, status, message)
    }
    async fn bytes(&self, response: reqwest::Response, maximum: usize) -> Result<Vec<u8>> {
        let mut response = self.status(response).await?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| e.without_url())? {
            ensure!(
                bytes.len().saturating_add(chunk.len()) <= maximum,
                "DGX response too large"
            );
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
    pub(crate) async fn ready(&self) -> Result<()> {
        let response = self
            .request(reqwest::Method::GET, "health")?
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| e.without_url())?;
        let value: serde_json::Value = serde_json::from_slice(&self.bytes(response, 65536).await?)?;
        if self.stage == "inpainting" && self.options.inpainting_model == QWEN_INPAINT_MODEL {
            if value["capabilities"]["configurable_models"] == true {
                ensure!(value["protocol"] == 1 && value["service"] == "qwen-inpaint"
                    && value["device"] == "cuda" && value["capabilities"]["inpainting_options"] == true,
                    "ComfyUI 인페인트 API가 호환되지 않습니다.");
                let wanted = &self.options.qwen_inpainting;
                let mut selection = serde_json::Map::new();
                for (key, name) in [("model", &wanted.model), ("text_encoder", &wanted.text_encoder), ("vae", &wanted.vae)] {
                    if !name.is_empty() && !(key == "model" && name == QWEN_INPAINT_MODEL)
                        && value["runtime"]["models"][key].as_str() != Some(name.as_str()) {
                        selection.insert(key.into(), json!(name));
                    }
                }
                if !selection.is_empty() {
                    let response = self.request(reqwest::Method::POST,"v1/models/select")?
                        .json(&selection).send().await.map_err(|e| e.without_url())?;
                    let selected: serde_json::Value = serde_json::from_slice(&self.bytes(response,65536).await?)?;
                    ensure!(selected["ready"] == true,"ComfyUI 모델 사전 로드 실패");
                    for (key, name) in selection.iter() {
                        ensure!(selected["models"][key] == *name,"ComfyUI 선택 모델 응답 불일치");
                    }
                } else {
                    ensure!(value["ready"] == true && value["model_loaded"] == true,
                            "ComfyUI 인페인트가 준비되지 않았습니다.");
                }
            } else { validate_qwen_health(&value)?; }

        } else if self.stage == "ocr" || self.stage == "detection" {
            ensure!(value["protocol"] == 1 && value["service"] == "koharu-vision"
                    && value["ready"] == true && value["model_loaded"] == true,"Vision 서버 준비 상태를 확인하세요.");
            let (field, name) = if self.stage == "ocr" { ("ocr_model", &self.options.ocr.model) }
                               else { ("model", &self.options.detection.model) };
            ensure!(name.is_empty() || value[field].as_str() == Some(name.as_str()),
                    "선택한 모델이 Vision 서버에 로드되어 있지 않습니다. 모델 경로 설정 후 서버를 재시작하세요.");
            if self.stage == "ocr" { ensure!(value["ocr_model_loaded"] == true
                && value["capabilities"]["hayai_ocr_options"] == true,"OCR 옵션 지원을 확인하세요."); }
        } else { validate_health(&value, &self.stage)?; }
        let seed = if self.options.inpainting_model == QWEN_INPAINT_MODEL {
            self.options.qwen_inpainting.seed
        } else { self.options.inpainting.seed };
        if self.stage == "inpainting" && seed == -1 {
            ensure!(value["capabilities"]["random_seed"] == true,
                    "인페인트 서버를 업데이트하세요. -1 랜덤 시드를 지원하지 않습니다.");
        }
        Ok(())
    }
    pub(crate) async fn detect(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        let options = json!({"model":self.options.detection.model,"text":thresholds.text,"onomatopoeia":thresholds.onomatopoeia,"bubble":thresholds.bubble,"panel":thresholds.panel});
        let response = self
            .request(reqwest::Method::POST, "v1/detect")?
            .multipart(
                Form::new()
                    .part("image", image_part(image)?)
                    .text("options", options.to_string()),
            )
            .send()
            .await
            .map_err(|e| e.without_url())?;
        let wire: DetectionResponse =
            serde_json::from_slice(&self.bytes(response, 64 * 1024 * 1024).await?)?;
        ensure!(
            wire.protocol == 1 && (wire.image_width, wire.image_height) == image.dimensions(),
            "Detection protocol/dimensions mismatch"
        );
        ensure!(wire.detections.len() <= 160, "too many detections");
        let mut output = Vec::new();
        let mut mask_pixels = 0usize;
        for d in wire.detections {
            let labels = ["text", "onomatopoeia", "bubble", "panel"];
            ensure!(
                d.label_id < 4
                    && d.label == labels[d.label_id]
                    && d.score.is_finite()
                    && (0.0..=1.0).contains(&d.score),
                "invalid detection label/confidence"
            );
            let [left, top, right, bottom] = d.bbox;
            ensure!(
                d.bbox.iter().all(|x| x.is_finite())
                    && left >= 0.0
                    && top >= 0.0
                    && right > left
                    && bottom > top
                    && right <= image.width() as f32
                    && bottom <= image.height() as f32,
                "invalid detection bounds"
            );
            let mask = decode(d.mask, image.width(), image.height())?;
            mask_pixels = mask_pixels
                .checked_add(mask.pixels.len())
                .context("mask overflow")?;
            ensure!(
                mask_pixels <= 256 * 1024 * 1024,
                "aggregate masks too large"
            );
            let area = mask.pixels.iter().filter(|p| **p != 0).count() as u32;
            output.push(KoharuLayoutDetection {
                label_id: d.label_id,
                label: d.label,
                score: d.score,
                bbox: d.bbox,
                area,
                mask,
            });
        }
        Ok(KoharuLayoutDetections {
            image_width: wire.image_width,
            image_height: wire.image_height,
            detections: output,
        })
    }
    pub(crate) async fn ocr(&self, image: &DynamicImage) -> Result<String> {
        let response = self
            .request(reqwest::Method::POST, "v1/ocr")?
            .multipart(Form::new().part("image", image_part(image)?)
                .text("options", serde_json::to_string(&self.options.ocr)?))
            .send()
            .await
            .map_err(|e| e.without_url())?;
        let value: OcrResponse = serde_json::from_slice(&self.bytes(response, 1024 * 1024).await?)?;
        ensure!(
            value.protocol == 1 && (self.options.ocr.model.is_empty() || value.model == self.options.ocr.model)
                && !value.revision.is_empty()
                && value.text.len() <= 65536 && !value.text.contains('\0'),
            "invalid OCR response"
        );
        Ok(value.text)
    }
    pub(crate) async fn inpaint(
        &self,
        image: &DynamicImage,
        mask: &GrayImage,
    ) -> Result<DynamicImage> {
        ensure!(
            image.dimensions() == mask.dimensions(),
            "image/mask dimensions mismatch"
        );
        let response = self
            .request(reqwest::Method::POST, "v1/inpaint")?
            .multipart(
                Form::new()
                    .part("image", image_part(image)?)
                    .part("mask", image_part(&DynamicImage::ImageLuma8(mask.clone()))?)
                    .text("options", if self.options.inpainting_model == QWEN_INPAINT_MODEL {
                        serde_json::to_string(&self.options.qwen_inpainting)?
                    } else { serde_json::to_string(&self.options.inpainting)? }),
            )
            .send()
            .await
            .map_err(|e| e.without_url())?;
        let response = self.status(response).await?;
        ensure!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.split(';').next() == Some("image/png")),
            "Inpaint service must return PNG"
        );
        let bytes = self.bytes(response, 256 * 1024 * 1024).await?;
        let dimensions =
            image::ImageReader::with_format(Cursor::new(&bytes), image::ImageFormat::Png)
                .into_dimensions()?;
        ensure!(
            dimensions == image.dimensions(),
            "RORem result dimensions mismatch"
        );
        let result = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)?;
        ensure!(
            result.dimensions() == image.dimensions(),
            "RORem result dimensions mismatch"
        );
        Ok(result)
    }
}
fn validate_qwen_health(value: &serde_json::Value) -> Result<()> {
    ensure!(value["protocol"] == 1 && value["service"] == "qwen-inpaint"
            && value["model"] == QWEN_INPAINT_MODEL && value["ready"] == true
            && value["model_loaded"] == true && value["device"] == "cuda"
            && value["capabilities"]["qwen_image21_turbo"] == true
            && value["capabilities"]["reference_mask_edit"] == true
            && value["capabilities"]["model_revision"] == QWEN_INPAINT_REVISION
            && value["capabilities"]["inpainting_options"] == true
            && value["runtime"]["warmup_complete"] == true,
            "Qwen 2.1 Turbo 서버·리비전·워밍업을 확인하세요. 기본 포트는 8765입니다.");
    Ok(())
}
fn validate_health(value: &serde_json::Value, stage: &str) -> Result<()> {
    let service = if stage == "inpainting" {
        "rorem-inpaint"
    } else {
        "koharu-vision"
    };
    ensure!(
        value["protocol"] == 1
            && value["ready"] == true
            && value["model_loaded"] == true
            && value["service"] == service,
        "DGX service not ready or incompatible"
    );
    if stage == "inpainting" {
        ensure!(value["model"] == "RORem-4S", "RORem model mismatch");
        ensure!(value["capabilities"]["inpainting_options"] == true,
                "RORem 서버를 업데이트하세요. Custom 추론 옵션을 지원하지 않습니다.");
    }
    if stage == "ocr" {
        ensure!(value["capabilities"]["hayai_ocr"] == true
                && value["capabilities"]["hayai_ocr_options"] == true
                && value["ocr_model_loaded"] == true && value["ocr_model"] == HAYAI_MODEL
                && value["ocr_revision"] == HAYAI_REVISION,
                "vision 서버를 업데이트하세요. Hayai OCR 모델 또는 고급 옵션을 지원하지 않습니다.");
    }
    Ok(())
}
#[derive(Deserialize)]
struct DetectionResponse {
    protocol: u32,
    image_width: u32,
    image_height: u32,
    detections: Vec<WireDetection>,
}
#[derive(Deserialize)]
struct WireDetection {
    label_id: usize,
    label: String,
    score: f32,
    bbox: [f32; 4],
    mask: MaskRle,
}
const HAYAI_MODEL: &str = "JustANormalTinkerer/hayai-ocr-v2";
const HAYAI_REVISION: &str = "4a4ce477c9a8841f208b94e1d9ed5c0938965e05";

#[derive(Deserialize)]
struct OcrResponse {
    protocol: u32,
    text: String,
    model: String,
    revision: String,
}
#[derive(Deserialize)]
struct MaskRle {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    runs: Vec<u32>,
}
fn decode(m: MaskRle, width: u32, height: u32) -> Result<KoharuLayoutMask> {
    ensure!(
        m.width > 0
            && m.height > 0
            && m.x.checked_add(m.width).is_some_and(|x| x <= width)
            && m.y.checked_add(m.height).is_some_and(|y| y <= height),
        "mask outside image"
    );
    let size = (m.width as usize)
        .checked_mul(m.height as usize)
        .context("mask overflow")?;
    ensure!(
        size <= 64 * 1024 * 1024 && m.runs.len() <= size + 1,
        "mask too large"
    );
    let mut pixels = Vec::with_capacity(size);
    let mut value = 0;
    for run in m.runs {
        let n = pixels
            .len()
            .checked_add(run as usize)
            .context("RLE overflow")?;
        ensure!(n <= size, "invalid mask RLE");
        pixels.resize(n, value);
        value = if value == 0 { 1 } else { 0 };
    }
    ensure!(pixels.len() == size, "incomplete mask RLE");
    Ok(KoharuLayoutMask {
        x: m.x,
        y: m.y,
        width: m.width,
        height: m.height,
        pixels,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rollback_config_backup_keeps_original_without_overwrite() {
        let id=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("koharu-qwen-backup-{}-{id}",std::process::id()));
        std::fs::create_dir(&root).unwrap();let path=root.join("config.toml");
        std::fs::write(&path,b"[custom-models]\nenabled=true\n").unwrap();
        backup_pre_qwen_config(&path).unwrap();
        std::fs::write(&path,b"new settings").unwrap();backup_pre_qwen_config(&path).unwrap();
        assert_eq!(std::fs::read(root.join("config.before-qwen-0.6.0.toml")).unwrap(),b"[custom-models]\nenabled=true\n");
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn selectable_model_names_are_persisted_and_validated() {
        let raw = r#"{"detection":{"model":"local/layout-v2"},"ocr":{"model":"local/hayai-v3"},"qwen_inpainting":{"model":"custom-turbo.safetensors","text_encoder":"encoder.safetensors","vae":"vae.safetensors"}}"#;
        let options: ModelOptions = serde_json::from_str(raw).unwrap();
        options.validate().unwrap();
        let saved = serde_json::to_value(&options).unwrap();
        assert_eq!(saved["ocr"]["model"],"local/hayai-v3");
        assert_eq!(saved["qwen_inpainting"]["model"],"custom-turbo.safetensors");
        let mut invalid=options;invalid.ocr.model="bad\nname".into();
        assert!(invalid.validate().is_err());
    }
    fn qwen_health() -> serde_json::Value {
        json!({"protocol":1,"service":"qwen-inpaint","model":QWEN_INPAINT_MODEL,
            "ready":true,"model_loaded":true,"device":"cuda",
            "runtime":{"warmup_complete":true},
            "capabilities":{"qwen_image21_turbo":true,"reference_mask_edit":true,
                "model_revision":QWEN_INPAINT_REVISION,"inpainting_options":true,"random_seed":true}})
    }
    #[test]
    fn qwen_defaults_migration_health_and_4096_limit() {
        let base = ModelOptions::default();
        assert_eq!(base.inpainting_model, QWEN_INPAINT_MODEL);
        assert_eq!(base.qwen_inpainting.max_side,2048);
        assert!(base.qwen_inpainting.use_kv_cache);
        let legacy: ModelOptions = serde_json::from_str(r#"{"inpainting":{"seed":7,"max_side":512}}"#).unwrap();
        assert_eq!(legacy.inpainting_model,"RORem-4S");
        assert_eq!(legacy.inpainting.seed,7);
        let mut limit=base.clone();limit.qwen_inpainting.max_side=4096;limit.validate().unwrap();
        limit.qwen_inpainting.max_side=4097;assert!(limit.validate().is_err());
        let good=qwen_health();validate_qwen_health(&good).unwrap();
        for (key,value) in [("service",json!("rorem-inpaint")),("model",json!("Qwen-Image-Edit-2511")),
                             ("device",json!("cpu")),("runtime",json!({"warmup_complete":false})),
                             ("capabilities",json!({"qwen_image21_turbo":true,"model_revision":"main"}))] {
            let mut bad=good.clone();bad[key]=value;assert!(validate_qwen_health(&bad).is_err());
        }
    }
    #[tokio::test]
    async fn qwen_multipart_has_only_supported_turbo_options() {
        let image=DynamicImage::new_rgb8(2,2);let mask=GrayImage::new(2,2);
        let (url,task)=mock_server(200,"image/png",png(&image).unwrap()).await;
        let mut remote=client(url,"inpainting");remote.options.inpainting_model=QWEN_INPAINT_MODEL.into();
        remote.options.qwen_inpainting.max_side=4096;remote.options.qwen_inpainting.seed=-1;
        remote.inpaint(&image,&mask).await.unwrap();
        let request=task.await.unwrap();
        assert!(request.contains("\"max_side\":4096") && request.contains("\"seed\":-1"));
        assert!(request.contains("\"use_kv_cache\":true"));
        for key in ["\"steps\"","\"strength\"","\"negative_prompt\"","\"guidance_scale\""] {
            assert!(!request.contains(key));
        }
        let (url,task)=mock_server(200,"application/json",qwen_health().to_string().into_bytes()).await;
        let mut remote=client(url,"inpainting");remote.options.inpainting_model=QWEN_INPAINT_MODEL.into();
        remote.ready().await.unwrap();task.await.unwrap();
    }
    #[test]
    fn token_normalization_and_validation() {
        assert_eq!(token("  Bearer Bearer abc123  ").unwrap(), "abc123");
        assert_eq!(token("  ").unwrap(), "");
        for value in ["Bearer ", "a\nb", "a\rb", "a b", "한글"] {
            assert!(token(value).is_err());
        }
    }
    #[test]
    fn url_contract() {
        assert!(origin("http://127.0.0.1:8766/").is_ok());
        for url in [
            "file:///x",
            "http://user:pass@host",
            "http://host/v1",
            "http://host/?secret=x",
        ] {
            assert!(origin(url).is_err());
        }
    }
    #[test]
    fn mask_contract() {
        let mask = decode(
            MaskRle {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
                runs: vec![1, 2, 1],
            },
            2,
            2,
        )
        .unwrap();
        assert_eq!(mask.pixels, vec![0, 1, 1, 0]);
        for runs in [vec![5], vec![1], vec![0, 5]] {
            assert!(
                decode(
                    MaskRle {
                        x: 0,
                        y: 0,
                        width: 2,
                        height: 2,
                        runs
                    },
                    2,
                    2
                )
                .is_err()
            );
        }
    }
    #[test]
    fn custom_defaults_and_option_validation() {
        let base = ModelOptions::default();
        assert!(base.inpainting_tiled);
        assert_eq!(base.ocr.max_new_tokens, 256);
        assert_eq!(base.ocr.num_beams, 1);
        assert_eq!(base.ocr.repetition_penalty, 1.0);
        assert_eq!(base.ocr.length_penalty, 1.0);
        assert!(base.ocr.early_stopping);
        let mut random = base.clone(); random.inpainting.seed = -1;
        random.validate().unwrap();
        for seed in [-2,2147483648] {
            let mut invalid = base.clone(); invalid.inpainting.seed = seed;
            assert!(invalid.validate().is_err());
        }
        for (tokens,beams,penalty,length) in [(0,1,1.0,1.0),(2049,1,1.0,1.0),(256,5,1.0,1.0),
                                             (256,1,f32::NAN,1.0),(256,1,1.0,0.0)] {
            let mut invalid = base.clone();
            invalid.ocr.max_new_tokens=tokens; invalid.ocr.num_beams=beams;
            invalid.ocr.repetition_penalty=penalty; invalid.ocr.length_penalty=length;
            assert!(invalid.validate().is_err());
        }
        assert_eq!(base.inpainting.steps, 4);
        assert_eq!(base.inpainting.prompt, koharu_ml::rorem_mixed::DEFAULT_PROMPT);
        assert_eq!(base.inpainting.max_side, koharu_ml::rorem_mixed::RoremMixedOptions::default().resolution);
        base.validate().unwrap();
        let mut invalid = base.clone(); invalid.detection.text = f32::NAN;
        assert!(invalid.validate().is_err());
        let mut invalid = base.clone(); invalid.inpainting.strength = 0.1;
        assert!(invalid.validate().is_err());
        let mut invalid = base.clone(); invalid.inpainting.mask_dilation = 33;
        assert!(invalid.validate().is_err());
        let mut invalid = base; invalid.inpainting.prompt = "bad\0prompt".into();
        assert!(invalid.validate().is_err());
        let initial: Settings = serde_json::from_str("{}").unwrap();
        assert!(initial.options.is_none());
    }
    #[tokio::test]
    async fn selected_stages_optional_tokens_and_session_retention() {
        let save = |vision: Option<&str>, inpainting: Option<&str>, timeout| {
            save_custom_models(&json!({
            "settings":{"enabled":true,"timeout":timeout},"vision_token":vision,"inpainting_token":inpainting}).to_string())
        };
        save(Some(""), Some(""), 600).unwrap();
        assert_eq!(Remote::configured("detection").unwrap().unwrap().token, "");
        save(Some(" Bearer v-token "), Some("r-token"), 600).unwrap();
        let snapshot = save(None, None, 900).unwrap();
        assert!(!snapshot.contains("v-token") && !snapshot.contains("r-token"));
        let configured = Remote::configured("detection").unwrap().unwrap();
        assert_eq!(configured.token, "v-token");
        assert!(save(Some("bad\ntoken"), None, 900).is_err());
        assert_eq!(
            Remote::configured("detection").unwrap().unwrap().token,
            "v-token"
        );
        let persisted = serde_json::to_value(settings().unwrap()).unwrap();
        // Public model options persist; authentication fields and session values do not.
        for key in ["token", "vision_token", "inpainting_token"] {
            assert!(persisted.get(key).is_none());
        }
        assert_eq!(persisted["options"]["inpainting"]["steps"], 4);
        let serialized = persisted.to_string();
        assert!(!serialized.contains("v-token") && !serialized.contains("r-token"));
        // Disabled stages may have no URL and never contact an absent vision server.
        let body = qwen_health().to_string();
        let (url, task) = mock_server(200, "application/json", body.into_bytes()).await;
        save_custom_models(
            &json!({"settings":{"enabled":true,
            "detection_enabled":false,"ocr_enabled":false,"inpainting_enabled":true,
            "detection_url":"","ocr_url":"","inpainting_url":url.as_str(),
            "options":{"inpainting_model":"RORem-4S"}},
            "vision_token":"","inpainting_token":""})
            .to_string(),
        )
        .unwrap();
        assert!(Remote::configured("detection").unwrap().is_none());
        assert!(Remote::configured("ocr").unwrap().is_none());
        assert!(check_custom_models().await.unwrap().contains("1개"));
        assert!(
            !task
                .await
                .unwrap()
                .to_lowercase()
                .contains("authorization:")
        );
        save_custom_models(
            &json!({"settings":{"enabled":false},
            "vision_token":null,"inpainting_token":null})
            .to_string(),
        )
        .unwrap();
        assert!(check_custom_models().await.unwrap().contains("없습니다"));
    }
    #[test]
    fn stage_switches_route_independently() {
        for bits in 0..8 {
            let mut settings = Settings {
                enabled: true,
                detection_enabled: bits & 1 != 0,
                ocr_enabled: bits & 2 != 0,
                inpainting_enabled: bits & 4 != 0,
                ..Settings::default()
            };
            for (stage, mask) in [("detection", 1), ("ocr", 2), ("inpainting", 4)] {
                assert_eq!(settings.stage_enabled(stage).unwrap(), bits & mask != 0);
            }
            settings.enabled = false;
            for stage in ["detection", "ocr", "inpainting"] {
                assert!(!settings.stage_enabled(stage).unwrap());
            }
            assert!(settings.stage_enabled("other").is_err());
        }
    }
    #[test]
    fn health_contract() {
        let good = json!({"protocol":1,"service":"rorem-inpaint","ready":true,"model_loaded":true,"model":"RORem-4S","capabilities":{"inpainting_options":true}});
        validate_health(&good, "inpainting").unwrap();
        for (key, value) in [
            ("ready", json!(false)),
            ("model_loaded", json!(false)),
            ("protocol", json!(2)),
            ("service", json!("other")),
            ("model", json!("other")),
        ] {
            let mut bad = good.clone();
            bad[key] = value;
            assert!(validate_health(&bad, "inpainting").is_err());
        }
        let hayai = json!({"protocol":1,"service":"koharu-vision","ready":true,"model_loaded":true,
            "ocr_model_loaded":true,"ocr_model":HAYAI_MODEL,"ocr_revision":HAYAI_REVISION,
            "capabilities":{"hayai_ocr":true,"hayai_ocr_options":true}});
        validate_health(&hayai, "ocr").unwrap();
        for (key, value) in [("ocr_model_loaded",json!(false)), ("ocr_model",json!("existing-Gemma")),
                             ("ocr_revision",json!("main")), ("capabilities",json!({"hayai_ocr":true}))] {
            let mut bad = hayai.clone();
            bad[key] = value;
            assert!(validate_health(&bad, "ocr").is_err());
        }
    }
    async fn mock_server(
        status: u16,
        mime: &str,
        body: Vec<u8>,
    ) -> (Url, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let mime = mime.to_owned();
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            loop {
                let mut buffer = [0u8; 4096];
                let n = stream.read(&mut buffer).await.unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..n]);
                if request.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let header = format!(
                "HTTP/1.1 {status} Test\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();
            String::from_utf8_lossy(&request).into_owned()
        });
        (url, task)
    }
    fn client(url: Url, stage: &str) -> Remote {
        Remote {
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(1))
                .build()
                .unwrap(),
            url,
            token: token("Bearer abc123").unwrap(),
            stage: stage.into(),
            options: ModelOptions { inpainting_model: legacy_inpaint_model(), ..Default::default() },
        }
    }
    #[tokio::test]
    async fn actual_health_header_and_401() {
        let body = json!({"protocol":1,"service":"koharu-vision","ready":true,"model_loaded":true,
                          "model":"mayocream/koharu-layout-rfdetr-seg-2xl-1152"})
            .to_string();
        let (url, task) = mock_server(200, "application/json", body.into_bytes()).await;
        client(url, "detection").ready().await.unwrap();
        let request = task.await.unwrap().to_lowercase();
        assert!(
            request.contains("authorization: bearer abc123\r\n")
                && !request.contains("bearer bearer")
        );
        let (url, task) = mock_server(401, "application/json", b"{}".to_vec()).await;
        let error = client(url, "detection")
            .ready()
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("401") && !error.contains("abc123"));
        task.await.unwrap();
    }
    #[tokio::test]
    async fn png_and_dimensions_contract() {
        let image = DynamicImage::new_rgb8(2, 2);
        let mask = GrayImage::new(2, 2);
        let (url, task) = mock_server(200, "image/png", png(&image).unwrap()).await;
        let mut remote = client(url, "inpainting");
        remote.token.clear();
        assert_eq!(
            remote.inpaint(&image, &mask).await.unwrap().dimensions(),
            (2, 2)
        );
        assert!(
            !task
                .await
                .unwrap()
                .to_lowercase()
                .contains("authorization:")
        );
        let (url, task) = mock_server(200, "image/jpeg", png(&image).unwrap()).await;
        assert!(
            client(url, "inpainting")
                .inpaint(&image, &mask)
                .await
                .is_err()
        );
        task.await.unwrap();
        let (url, task) = mock_server(
            200,
            "image/png",
            png(&DynamicImage::new_rgb8(3, 3)).unwrap(),
        )
        .await;
        assert!(
            client(url, "inpainting")
                .inpaint(&image, &mask)
                .await
                .is_err()
        );
        task.await.unwrap();
    }
    #[tokio::test]
    async fn inpaint_http_errors_precede_png_check_and_hide_untrusted_text() {
        let image = DynamicImage::new_rgb8(2, 2);
        let mask = GrayImage::new(2, 2);
        for (status, body, message) in [
            (413, json!({"detail":{"code":"upload_too_large","max_upload_bytes":134217728}}), "134217728 bytes per file"),
            (413, json!({"detail":"Upload exceeds 32 MiB"}), "server size limit"),
            (500, json!({"detail":{"code":"cuda_out_of_memory","type":"OutOfMemoryError"}}), "cuda_out_of_memory"),
            (422, json!({"detail":"secret-token-abc123"}), "invalid image, mask or options"),
            (500, json!({"detail":{"code":"secret_token_abc123"}}), "server request failed"),
        ] {
            let (url, task) = mock_server(status, "application/json", body.to_string().into_bytes()).await;
            let error = client(url, "inpainting").inpaint(&image, &mask).await.unwrap_err().to_string();
            assert!(error.contains(&status.to_string()) && error.contains(message), "{error}");
            assert!(!error.contains("must return PNG") && !error.contains("abc123"));
            task.await.unwrap();
        }
    }
    #[tokio::test]
    async fn custom_options_are_sent_in_actual_multipart_requests() {
        let image = DynamicImage::new_rgb8(2, 2);
        let mask = GrayImage::new(2, 2);
        let (url, task) = mock_server(200, "image/png", png(&image).unwrap()).await;
        let mut remote = client(url, "inpainting");
        remote.options.inpainting.prompt = "Keep the original line art".into();
        remote.options.inpainting.seed = -1;
        remote.inpaint(&image, &mask).await.unwrap();
        let request = task.await.unwrap();
        assert!(request.contains("name=\"options\""));
        assert!(request.contains("Keep the original line art") && request.contains("\"seed\":-1"));
        assert!(request.contains("\"max_side\":512"));
        let response = json!({"protocol":1,"text":"read", "model":HAYAI_MODEL,"revision":HAYAI_REVISION});
        let (url, task) = mock_server(200, "application/json", response.to_string().into_bytes()).await;
        let mut remote = client(url, "ocr");
        remote.options.ocr.max_new_tokens = 512;
        remote.options.ocr.num_beams = 3;
        remote.options.ocr.repetition_penalty = 1.2;
        remote.options.ocr.length_penalty = 0.7;
        remote.options.ocr.early_stopping = false;
        assert_eq!(remote.ocr(&image).await.unwrap(), "read");
        let request = task.await.unwrap();
        assert!(request.contains("name=\"image\""));
        assert!(request.contains("name=\"options\""));
        assert!(request.contains("\"max_new_tokens\":512") && request.contains("\"num_beams\":3"));
        assert!(request.contains("\"repetition_penalty\":1.2") && request.contains("\"length_penalty\":0.7"));
        assert!(request.contains("\"early_stopping\":false"));
    }
    #[tokio::test]
    async fn random_seed_requires_server_support() {
        for supported in [false,true] {
            let body = json!({"protocol":1,"service":"rorem-inpaint","ready":true,
                "model_loaded":true,"model":"RORem-4S",
                "capabilities":{"inpainting_options":true,"random_seed":supported}});
            let (url,task)=mock_server(200,"application/json",body.to_string().into_bytes()).await;
            let mut remote=client(url,"inpainting"); remote.options.inpainting.seed=-1;
            assert_eq!(remote.ready().await.is_ok(),supported);
            task.await.unwrap();
        }
    }
    #[tokio::test]
    async fn inference_timeout_is_an_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let task = tokio::spawn(async move {
            let (_stream, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_millis(150)).await;
        });
        let mut remote = client(url, "ocr");
        remote.client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(30))
            .build()
            .unwrap();
        let error = remote.ocr(&DynamicImage::new_rgb8(2, 2)).await.unwrap_err();
        assert!(error.downcast_ref::<reqwest::Error>().unwrap().is_timeout());
        task.await.unwrap();
    }
    #[tokio::test]
    async fn ocr_rejects_wrong_model_and_missing_revision() {
        for (model, revision) in [("existing-Gemma", HAYAI_REVISION), (HAYAI_MODEL, "")] {
            let body = json!({"protocol":1,"text":"read","model":model,"revision":revision});
            let (url, task) = mock_server(200,"application/json",body.to_string().into_bytes()).await;
            assert!(client(url,"ocr").ocr(&DynamicImage::new_rgb8(2,2)).await.is_err());
            task.await.unwrap();
        }
    }
    #[tokio::test]
    async fn detection_rejects_wrong_coordinates_and_malformed_rle() {
        let image = DynamicImage::new_rgb8(2, 2);
        let mut wire = json!({"protocol":1,"image_width":3,"image_height":2,"detections":[]});
        let thresholds = KoharuLayoutThresholds {
            text: 0.25,
            onomatopoeia: 0.20,
            bubble: 0.5,
            panel: 0.5,
        };
        let (url, task) = mock_server(200, "application/json", wire.to_string().into_bytes()).await;
        assert!(
            client(url, "detection")
                .detect(&image, thresholds)
                .await
                .is_err()
        );
        task.await.unwrap();
        wire["image_width"] = json!(2);
        wire["detections"] = json!([{"label_id":0,"label":"text","score":0.9,
            "bbox":[0,0,2,2],"mask":{"x":0,"y":0,"width":2,"height":2,"runs":[1]}}]);
        let (url, task) = mock_server(200, "application/json", wire.to_string().into_bytes()).await;
        assert!(
            client(url, "detection")
                .detect(&image, thresholds)
                .await
                .is_err()
        );
        task.await.unwrap();
    }
}
