"""Maintainer tool: produce a small, reviewable patch from the pinned pristine source."""

from pathlib import Path
import difflib, sys

ROOT = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).resolve()
changes = {}


def replace(path, old, new):
    text = changes.get(path, (source / path).read_text(encoding="utf-8"))
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one source boundary: {path}: {old[:60]}")
    changes[path] = text.replace(old, new, 1)


base = "crates/koharu-pipeline/"
replace(
    base + "Cargo.toml",
    "\nrayon = { workspace = true }",
    "\nrayon = { workspace = true }\nreqwest = { workspace = true }\nserde_json = { workspace = true }",
)
replace(
    base + "src/lib.rs",
    "mod resources;",
    "mod resources;\nmod remote;\npub use remote::{get_custom_models, save_custom_models, check_custom_models};",
)
d = base + "src/stages/detection.rs"
replace(
    d,
    "network: Arc<Mutex<KoharuLayoutRFDetrSeg2XL>>,",
    "network: Option<Arc<Mutex<KoharuLayoutRFDetrSeg2XL>>>,\n    remote: Option<crate::remote::Remote>,",
)
replace(
    d,
    "        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = config;",
    """        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = config;
        if let Some(remote) = crate::remote::Remote::configured("detection")? {
            remote.ready().await?;
            let thresholds = remote.detection_thresholds();
            return Ok(Self { network: None, remote: Some(remote), thresholds });
        }""",
)
replace(
    d,
    "network: Arc::new(Mutex::new(network)),",
    "network: Some(Arc::new(Mutex::new(network))),\n            remote: None,",
)
replace(
    d,
    "        let network = self.network.clone();",
    """        if let Some(remote) = &self.remote {
            return remote.detect(&image, self.thresholds).await;
        }
        let network = self.network.as_ref().context("local detector not loaded")?.clone();""",
)
replace(
    d,
    "        MODEL_NAME\n",
    """        if crate::remote::enabled("detection") { return "dgx-detection"; }
        MODEL_NAME
""",
)
o = base + "src/stages/ocr.rs"
replace(o, "enum Model {", "enum Model {\n    Remote(crate::remote::Remote),")
replace(
    o,
    "        match config {\n",
    """        if let Some(remote) = crate::remote::Remote::configured("ocr")? {
            remote.ready().await?; return Ok(Self::Remote(remote));
        }
        match config {
""",
)
replace(
    o,
    "        let model_name = match self {",
    '        let model_name = match self {\n            Self::Remote(_) => "dgx-ocr",',
)
replace(
    o,
    "        let results = match self {",
    """        let results = match self {
            Self::Remote(remote) => {
                let mut results = Vec::new();
                for target in targets {
                    let text = normalize_ocr_text(remote.ocr(&target.image).await?);
                    results.push(OcrResult { content: target.content, region: target.region,
                        geometry: target.geometry, previous: target.previous, text });
                }
                results
            },""",
)
replace(
    o,
    "        match self.config {",
    """        if crate::remote::enabled("ocr") { return "dgx-ocr"; }
        match self.config {""",
)
i = base + "src/stages/inpainting.rs"
replace(i, "enum Model {", "enum Model {\n    Remote(crate::remote::Remote),")
replace(
    i,
    "        match config {\n",
    """        if let Some(remote) = crate::remote::Remote::configured("inpainting")? {
            remote.ready().await?; return Ok(Self::Remote(remote));
        }
        match config {
""",
)
replace(
    i,
    "        let (model_name, image) = match self {",
    """        let (model_name, image) = match self {
            Self::Remote(remote) => ("dgx-rorem-4s", inpaint_remote(prepared, remote).await?),""",
)
replace(
    i,
    "        match self.config {",
    """        if crate::remote::enabled("inpainting") { return "dgx-rorem-4s"; }
        match self.config {""",
)
replace(i, "async fn prepare(input:", 'async fn inpaint_remote(prepared: InpaintInput, remote: &crate::remote::Remote) -> Result<DynamicImage> {\n    if !remote.inpainting_tiled() {\n        return remote.inpaint(&prepared.image, &prepared.mask).await;\n    }\n    let (mut output, pending_mask, labels, tiles) = tokio_rayon::spawn(move || -> Result<_> {\n        let mut output = prepared.image.to_rgb8();\n        let mut pending = prepared.mask.clone();\n        fill_uniform_regions(&mut output, &mut pending, &prepared.text_mask, &prepared.flat_fill_regions);\n        let (labels, tiles) = inpaint_tiles(&pending);\n        Ok((output, pending, labels, tiles))\n    }).await?;\n    for tile in &tiles {\n        let [left, top, right, bottom] = tile.crop;\n        let crop = image::imageops::crop_imm(&output, left, top, right - left, bottom - top).to_image();\n        let mask = crop_tile_mask(&pending_mask, tile);\n        let generated = remote.inpaint(&DynamicImage::ImageRgb8(crop), &mask).await?.to_rgb8();\n        composite_generated(&mut output, &labels, tile, &generated);\n    }\n    Ok(DynamicImage::ImageRgb8(output))\n}\n\nasync fn prepare(input:')
replace(i, "    #[test]\n    fn uniform_mask_background_is_filled_without_inference()", '    #[tokio::test]\n    async fn remote_inpainting_uses_bounded_regions_and_preserves_unmasked_pixels() {\n        use tokio::io::{AsyncReadExt, AsyncWriteExt};\n        let source = Arc::new(DynamicImage::ImageRgb8(RgbImage::from_pixel(900, 900, Rgb([20, 40, 60]))));\n        let mut mask = GrayImage::new(900, 900);\n        mask.put_pixel(10, 10, Luma([255]));\n        mask.put_pixel(850, 850, Luma([255]));\n        let (_, tiles) = inpaint_tiles(&mask);\n        assert_eq!(tiles.len(), 2);\n        let sizes = tiles.iter().map(|t| (t.crop[2] - t.crop[0], t.crop[3] - t.crop[1])).collect::<Vec<_>>();\n        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();\n        let remote = crate::remote::Remote::test_client(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();\n        let server = tokio::spawn(async move {\n            for (width, height) in sizes {\n                assert!(width <= TILE_SIZE + 2 * TILE_CONTEXT && height <= TILE_SIZE + 2 * TILE_CONTEXT);\n                let (mut stream, _) = listener.accept().await.unwrap();\n                let mut request = Vec::new();\n                let mut chunk = [0u8; 8192];\n                loop {\n                    let n = stream.read(&mut chunk).await.unwrap();\n                    assert!(n > 0);\n                    request.extend_from_slice(&chunk[..n]);\n                    if let Some(header_end) = request.windows(4).position(|v| v == b"\\r\\n\\r\\n") {\n                        let headers = String::from_utf8_lossy(&request[..header_end]).to_lowercase();\n                        let size: usize = headers.lines().find_map(|line| line.strip_prefix("content-length:")).unwrap().trim().parse().unwrap();\n                        if request.len() >= header_end + 4 + size { break; }\n                    }\n                }\n                assert!(String::from_utf8_lossy(&request).contains("name=\\"options\\""));\n                let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([1, 2, 3])));\n                let mut png = Cursor::new(Vec::new());\n                image.write_to(&mut png, ImageFormat::Png).unwrap();\n                let bytes = png.into_inner();\n                stream.write_all(format!("HTTP/1.1 200 OK\\r\\nContent-Type: image/png\\r\\nContent-Length: {}\\r\\nConnection: close\\r\\n\\r\\n", bytes.len()).as_bytes()).await.unwrap();\n                stream.write_all(&bytes).await.unwrap();\n            }\n        });\n        let prepared = InpaintInput { image: source.clone(), original: source,\n            cleanup_entity: None, cleanup: None, mask: mask.clone(), text_mask: mask,\n            flat_fill_regions: Vec::new() };\n        let output = inpaint_remote(prepared, &remote).await.unwrap().to_rgb8();\n        assert_eq!(output.dimensions(), (900, 900));\n        assert_eq!(output.get_pixel(10, 10), &Rgb([1, 2, 3]));\n        assert_eq!(output.get_pixel(850, 850), &Rgb([1, 2, 3]));\n        assert_eq!(output.get_pixel(500, 500), &Rgb([20, 40, 60]));\n        server.await.unwrap();\n    }\n\n    #[test]\n    fn uniform_mask_background_is_filled_without_inference()')
a = "crates/koharu-app/src/commands/mod.rs"
replace(
    base + "src/pipeline.rs",
    "    pub fn subscribe_resources(&self)",
    """    /// Called under the app's processing admission lock, between jobs only.
    pub fn invalidate_custom_models(&self) {
        self.current.load_full().invalidate_custom_models();
    }

    pub fn subscribe_resources(&self)""",
)
replace(
    base + "src/stage_runner.rs",
    "    #[tracing::instrument(skip_all)]",
    """    pub(crate) fn invalidate_custom_models(&self) {
        for stage in [Stage::Detection, Stage::Ocr, Stage::Inpainting] {
            self.stages.unload(stage);
        }
    }

    #[tracing::instrument(skip_all)]""",
)
replace(a, "pub(crate) mod canvas;", "pub(crate) mod canvas;\npub(crate) mod custom;")
replace(
    a,
    "            preferences::get_preferences,",
    "            preferences::get_preferences,\n            custom::get_custom_models,\n            custom::save_custom_models,\n            custom::check_custom_models,",
)
f = "packages/koharu/components/preferences/SettingsPage.tsx"
replace(
    f,
    "import { PipelinePreferences }",
    "import { CustomPreferences } from '@/components/preferences/CustomPreferences'\nimport { PipelinePreferences }",
)
replace(f, "  ['pipeline', Cpu],", "  ['pipeline', Cpu],\n  ['custom', Cpu],")
replace(
    f,
    "{t(`settings.tabs.${id}`)}",
    "{id === 'custom' ? 'Custom' : t(`settings.tabs.${id}`)}",
)
replace(
    f,
    "            {tab === 'appearance' && <AppearancePreferences />}",
    "            {tab === 'custom' && <CustomPreferences onOpenTab={setTab} />}\n            {tab === 'appearance' && <AppearancePreferences />}",
)
patch = []
for name, updated in changes.items():
    original = (source / name).read_text(encoding="utf-8")
    patch.extend(
        difflib.unified_diff(
            original.splitlines(True),
            updated.splitlines(True),
            fromfile="a/" + name,
            tofile="b/" + name,
            n=3,
        )
    )
(ROOT / "patches/remote-models.patch").write_text("".join("\n" if line == " \n" else line for line in patch), encoding="utf-8")
print(
    f"{len(changes)} existing files changed; three new source modules. Native scene handling preserved."
)
