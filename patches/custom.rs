use super::{Error, processing::Processing};
use tauri::State;

#[tauri::command]
#[specta::specta]
pub(crate) fn get_custom_models() -> Result<String, Error> {
    Ok(koharu_pipeline::get_custom_models()?)
}
#[tauri::command]
#[specta::specta]
pub(crate) fn save_custom_models(
    config: String,
    processing: State<'_, Processing>,
    pipeline: State<'_, koharu_pipeline::Pipeline>,
) -> Result<String, Error> {
    // Same admission lock as process(): settings cannot change inside a running job.
    let stops = processing.stops.lock();
    if !stops.is_empty() {
        return Err(anyhow::anyhow!("작업 완료 후 Custom 설정을 변경하세요.").into());
    }
    let saved = koharu_pipeline::save_custom_models(&config)?;
    pipeline.invalidate_custom_models();
    Ok(saved)
}
#[tauri::command]
#[specta::specta]
pub(crate) async fn check_custom_models() -> Result<String, Error> {
    Ok(koharu_pipeline::check_custom_models().await?)
}
