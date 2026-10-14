"""SAM-TS geometry and tiling boundaries without weights or CUDA."""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from dgx.vision import text_sam


def test_release_rgb_prepare_preserves_channel_range_and_padding():
    image=Image.new('RGB',(300,150),(12,34,56))
    array,size=text_sam.prepare(image)
    assert size==(1024,512) and array.shape==(3,1024,1024)
    assert np.array_equal(array[:,20,20],[12,34,56])
    assert np.all(array[:,512:]==128)


def test_mask_restoration_discards_bottom_right_padding():
    logits=np.zeros((1024,1024),np.float32);logits[512:]=9
    result=text_sam.restore(logits,(1024,512),(300,150))
    assert result.shape==(150,300) and not result.any()
    logits[:512,:512]=1
    result=text_sam.restore(logits,(1024,512),(300,150))
    assert np.all(result[:,:150]==1) and not result[:,150:].any()


def test_overlapping_tiles_cover_page_and_restore_original_coordinates():
    pixels=np.zeros((704,608,3),np.uint8);pixels[120:640,245:260,0]=255
    image=Image.fromarray(pixels)
    calls=[];progress=[]
    def predict(crop):
        calls.append(crop.size)
        return np.asarray(crop,dtype=np.float32)[...,0]/255-0.5
    result=text_sam.segment_with(predict,image,mode='tiles',tile_size=512,overlap=128,progress=lambda *args:progress.append(args))
    assert np.array_equal(result,pixels[...,0]>0)
    assert len(calls)==4 and progress[-1]==(4,4)
    coverage=np.zeros(pixels.shape[:2],np.uint8)
    for x0,y0,x1,y1 in text_sam.tile_boxes(image.size,512,128):coverage[y0:y1,x0:x1]+=1
    assert coverage.min()>0


def test_full_page_threshold_and_no_tile_seam_at_constant_logits():
    image=Image.new('RGB',(800,700))
    def predict(crop):return np.full((crop.height,crop.width),0.3,np.float32)
    assert text_sam.segment_with(predict,image,mode='tiles',tile_size=512,overlap=128).all()
    assert not text_sam.segment_with(predict,image,threshold=0.4).any()


@pytest.mark.parametrize('kwargs',[{'mode':'other'},{'threshold':float('nan')},{'mode':'tiles','tile_size':512,'overlap':256}])
def test_invalid_options_fail_before_prediction(kwargs):
    calls=[]
    with pytest.raises(ValueError):text_sam.segment_with(lambda image:calls.append(image),Image.new('RGB',(700,700)),**kwargs)
    assert calls==[]


def test_model_source_uses_no_global_checkpoint_patch_or_pickle_loading():
    root=Path(text_sam.__file__).parent
    files=list((root/'_text_sam').glob('*.py'))+[root/'text_sam.py']
    for file in files:
        source=file.read_text()
        assert 'torch.load(' not in source
        assert 'torch.utils.checkpoint.checkpoint =' not in source


def test_tile_budget_is_bounded_before_inference():
    with pytest.raises(ValueError,match='Too many'):
        text_sam.tile_boxes((1536,8192),512,240)
    with pytest.raises(ValueError,match='pixel limit'):
        text_sam.tile_boxes((10000,10000))


@pytest.mark.skipif(os.environ.get('SAM_TS_META_TEST')!='1',reason='opt-in Torch meta architecture check')
def test_published_model_architecture_and_hr_forward_shapes_on_meta_device():
    import torch
    config={'variant':'vit_l_textseg','image_size':1024,'hierarchical_detection':False}
    with torch.device('meta'),torch.inference_mode():
        model=text_sam.build_model(config).eval().to('meta')
        state=model.state_dict()
        assert len(state)==728
        assert sum(t.numel() for t in state.values())==338934897
        output=model([{'image':torch.empty((3,1024,1024)),'original_size':(1024,1024)}],multimask_output=False)
    assert len(output)==6 and output[3].shape==(1,1,1024,1024)
