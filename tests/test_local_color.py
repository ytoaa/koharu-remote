"""Numerical background fixtures; no GPU/model quality claims."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('local_color',Path(__file__).resolve().parents[1]/'comfyui/koharu_nodes/local_color.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
match=module.local_color_match


def fixture():
    reference=np.full((64,96,3),0.5,np.float32)
    target=reference-0.025
    mask=np.zeros((64,96),np.float32);mask[16:48,20:76]=1
    return target,reference,mask


def test_background_offset_excludes_original_letters_and_preserves_inputs():
    target,reference,mask=fixture()
    reference[20:40,30:70]=1  # Original letters must not bias background offset.
    before=[a.copy() for a in (target,reference,mask)]
    result=match(target,reference,mask)
    assert np.allclose(result[24:36,35:65],0.5,atol=1e-5)
    for original,copy in zip((target,reference,mask),before):assert np.array_equal(original,copy)


def test_white_margin_does_not_override_gray_panel_offset():
    reference=np.full((96,128,3),0.95,np.float32);reference[20:76,20:108]=0.5
    target=reference.copy();target[20:76,20:108]-=0.03
    mask=np.zeros(reference.shape[:2],np.float32);mask[28:68,30:98]=1
    result=match(target,reference,mask,radius=48)
    assert np.allclose(result[40:56,45:83],0.5,atol=1e-4)
    assert np.allclose(result[0:10],0.95,atol=1e-4)


def test_rgb_mode_corrects_channels_while_luminance_preserves_channel_differences():
    target,reference,mask=fixture()
    target[:]=[0.46,0.48,0.49]
    rgb=match(target,reference,mask,mode='rgb')
    assert np.allclose(rgb,0.5,atol=1e-5)
    luminance=match(target,reference,mask,mode='luminance')
    assert np.allclose(luminance[...,1]-luminance[...,0],0.02,atol=1e-5)


def test_full_mask_no_samples_or_disabled_is_noop():
    target,reference,mask=fixture()
    assert np.array_equal(match(target,reference,np.ones_like(mask)),target)
    assert np.array_equal(match(target,reference,mask,strength=0),target)
    assert np.array_equal(match(target,reference,mask,max_shift=0),target)
    assert np.array_equal(match(target,reference+0.3,mask),target)


def test_shift_limit_and_strength_bound_correction():
    target,reference,mask=fixture()
    result=match(target,reference,mask,max_shift=0.01,strength=0.5)
    assert np.allclose(result-target,0.005,atol=1e-6)


def test_spatially_varying_offset_is_reduced_without_extrapolation_seams():
    target,reference,mask=fixture()
    offset=0.01+np.arange(target.shape[1],dtype=np.float32)*0.0004
    target=reference-offset[None,:,None]
    result=match(target,reference,mask)
    selected=mask>=0.5
    before=np.mean((target[selected]-reference[selected])**2)
    after=np.mean((result[selected]-reference[selected])**2)
    assert after<before*0.1
    # A smooth background should not acquire hard boundaries in the hole.
    assert np.max(np.abs(np.diff(result[32,25:71,0])))<0.002


def test_large_source_uses_same_grid_without_text_sample_leakage():
    target=np.full((768,1025,3),0.48,np.float32);reference=target+0.02
    mask=np.zeros(target.shape[:2],np.float32);mask[:,512]=1;reference[:,512]=1
    result=match(target,reference,mask)
    assert result.shape==target.shape and np.allclose(result,0.5,atol=1e-5)


@pytest.mark.parametrize('case',['shape','nan','mode','radius'])
def test_invalid_inputs_are_rejected(case):
    target,reference,mask=fixture();options={}
    if case=='shape':reference=reference[:20]
    if case=='nan':reference[0,0]=np.nan
    if case=='mode':options['mode']='other'
    if case=='radius':options['radius']=0
    with pytest.raises(ValueError):match(target,reference,mask,**options)
