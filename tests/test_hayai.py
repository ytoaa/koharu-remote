"""Crop geometry, release integrity and hallucination isolation without a GPU."""
import hashlib, io
import numpy as np
import pytest
from PIL import Image
from vision import hayai, hayai_files
import manage_vision as manage


@pytest.mark.parametrize("size,grid", [((1024,1024),(16,16)), ((900,120),(6,42)), ((4,4),(16,16))])
def test_official_crop_geometry_and_patch_padding(size, grid):
    pixels, mask, shape = hayai.preprocess(Image.new("RGB", size, "white"))
    assert tuple(shape[0]) == grid
    count = grid[0] * grid[1]
    assert pixels.shape == (1,256,768) and mask.shape == (1,256)
    np.testing.assert_array_equal(mask[0], [1]*count + [0]*(256-count))
    assert np.all(pixels[0,:count] == 1) and np.all(pixels[0,count:] == 0)


def test_rgb_patch_channel_and_pixel_order():
    rgb = np.zeros((256,256,3), dtype=np.uint8)
    rgb[0,0] = [255,128,0]
    rgb[0,16] = [0,0,255]
    pixels, _, _ = hayai.preprocess(Image.fromarray(rgb))
    np.testing.assert_allclose(pixels[0,0,:3], [1,128/255*2-1,-1], atol=1e-7)
    np.testing.assert_array_equal(pixels[0,1,:3], [-1,-1,1])


@pytest.mark.parametrize("text", ["どぅぅぅ", "あああああ", "こんにちは", "", "hello world"])
def test_short_effects_and_normal_text_survive(text):
    assert hayai.validate_result(text, 10) == text


@pytest.mark.parametrize("text,count", [("ど"+"ぅ"*2047,256), ("hello",256), ("あ"*40,40), ("abc"*12,36), ("bad\0text",8)])
def test_truncated_or_excessively_repeated_results_never_become_source_text(text,count):
    with pytest.raises(ValueError):
        hayai.validate_result(text,count)


def test_repetition_at_token_limit_has_repetition_code_for_bounded_retry():
    with pytest.raises(hayai.HayaiOutputError) as error:
        hayai.validate_result('abc'*100,256,256)
    assert error.value.code=='ocr_repetition_loop'
    with pytest.raises(hayai.HayaiOutputError) as error:
        hayai.validate_result('ordinary truncated sentence',256,256)
    assert error.value.code=='ocr_token_limit'


def test_ocr_rejection_logs_budget_and_crop_without_recognized_text(caplog):
    from vision.options import OcrOptions
    backend=hayai.Backend.__new__(hayai.Backend)
    backend._generate=lambda *args:('private unfinished sentence',512)
    with pytest.raises(hayai.HayaiOutputError):
        backend.ocr(Image.new('RGB',(1084,403)),OcrOptions(max_new_tokens=512),1024)
    assert 'tokens=512/512' in caplog.text and 'crop=1084x403' in caplog.text
    assert 'patch_budget=1024' in caplog.text and 'private unfinished sentence' not in caplog.text


def test_hash_mismatch_rejects_model_before_execution(tmp_path, monkeypatch):
    file = tmp_path / "modeling_hayai.py"
    original = b"safe"
    file.write_bytes(original)
    monkeypatch.setattr(hayai_files, "FILES", {file.name: hashlib.sha256(original).hexdigest()})
    assert hayai_files.validate(tmp_path) == tmp_path
    file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="pinned file hash"):
        hayai_files.validate(tmp_path)


def test_download_preserves_existing_models_and_incomplete_files(tmp_path, monkeypatch):
    content = b"pinned-model"
    hashes = {"weights": hashlib.sha256(content).hexdigest()}
    requests = []
    def fetch(url, **_):
        requests.append(url)
        return io.BytesIO(content)
    monkeypatch.setattr(manage.urllib.request, "urlopen", fetch)
    manage.download_files(tmp_path,"test","revision",hashes,{"weights":"pinned-url"})
    manage.download_files(tmp_path,"test","revision",hashes,{"weights":"pinned-url"})
    assert requests == ["pinned-url"]
    (tmp_path/"weights").write_bytes(b"existing")
    with pytest.raises(ValueError, match="refusing replacement"):
        manage.download_files(tmp_path,"test","revision",hashes,{"weights":"pinned-url"})
    assert (tmp_path/"weights").read_bytes() == b"existing"
    (tmp_path/"weights").unlink()
    (tmp_path/"weights.download").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Incomplete model download"):
        manage.download_files(tmp_path,"test","revision",hashes,{"weights":"pinned-url"})
    assert (tmp_path/"weights.download").read_bytes() == b"partial"


@pytest.mark.parametrize("dtype,custom", [("float32",False), ("bfloat16",False), ("float32",True)])
def test_checkpoint_matches_official_character_sequence(dtype, custom):
    import os
    from pathlib import Path
    folder = os.environ.get("HAYAI_MODEL_FIXTURE")
    upstream = os.environ.get("UPSTREAM_FIXTURE")
    if not folder or not upstream:
        pytest.skip("Opt-in real checkpoint check requires local weights and pinned Koharu fixture")
    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(2)
    folder = hayai_files.validate(folder)
    model = hayai.load_model(folder, device="cpu")
    model.to(dtype=getattr(torch, dtype))
    def unused_pooler_must_not_run(*_):
        pytest.fail("Hayai invoked the unused SigLIP pooled-output head")
    model.vision_encoder.head.register_forward_pre_hook(unused_pooler_must_not_run)
    image = Image.open(Path(upstream) / "crates/koharu-ml/benches/fixtures/ocr/title.png")
    backend = hayai.Backend.__new__(hayai.Backend)
    backend.device = "cpu"  # Test only: production constructor requires CUDA.
    backend.model = model
    backend.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    from vision.options import OcrOptions
    options = OcrOptions(max_new_tokens=24,num_beams=3,repetition_penalty=1.2,
                         length_penalty=0.7,early_stopping=False) if custom else OcrOptions()
    assert backend.ocr(image,options)["text"] == "対策委員会です!"


def test_truncation_guard_uses_the_requested_output_budget():
    assert hayai.validate_result("valid", 300, 512) == "valid"
    with pytest.raises(ValueError, match="token limit"):
        hayai.validate_result("cut", 24, 24)


@pytest.mark.parametrize('budget',[256,512,1024])
def test_expanded_patch_budget_preserves_padding_and_spatial_shape(budget):
    pixels,mask,shape=hayai.preprocess(Image.new('RGB',(1084,403),'white'),budget)
    rows,columns=shape[0]
    count=int(rows*columns)
    assert count<=budget
    assert pixels.shape==(1,budget,768) and mask.shape==(1,budget)
    assert np.all(mask[0,:count]==1) and np.all(mask[0,count:]==0)
    assert np.all(pixels[0,:count]==1) and np.all(pixels[0,count:]==0)
    if budget==1024:assert columns*16>800 and rows*16>300


def test_invalid_patch_budget_is_rejected_before_allocation():
    with pytest.raises(ValueError,match='max_patches'):hayai.preprocess(Image.new('RGB',(32,32)),8192)
