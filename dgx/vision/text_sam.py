"""Resident SAM-TS-L text pixels, with release preprocessing and optional tiles."""
import json, logging, math, time
from functools import partial
from pathlib import Path
import numpy as np
from PIL import Image
from .text_sam_files import REPO,REV,validate

LOG=logging.getLogger(__name__)
IMAGE_SIZE=1024
MAX_PIXELS=16*1024*1024
MAX_TILES=64


def prepare(image):
    """RGB 0..255, longest side 1024, bottom/right padding value 128."""
    image=image.convert('RGB')
    if min(image.size)<=0:raise ValueError('Empty text-mask image')
    scale=IMAGE_SIZE/max(image.size)
    resized_size=tuple(max(1,round(axis*scale)) for axis in image.size)
    resized=image.resize(resized_size,Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(IMAGE_SIZE,IMAGE_SIZE),(128,128,128));canvas.paste(resized,(0,0))
    array=np.asarray(canvas,dtype=np.float32).transpose(2,0,1).copy()
    return array,resized_size


def restore(values,resized_size,original_size):
    width,height=resized_size
    values=np.asarray(values,dtype=np.float32)
    if values.shape!=(IMAGE_SIZE,IMAGE_SIZE) or not np.isfinite(values).all():
        raise ValueError('Invalid SAM-TS high-resolution mask')
    # Nearest interpolation matches the published binary-mask restoration.
    return np.asarray(Image.fromarray(values[:height,:width]).resize(original_size,Image.Resampling.NEAREST),dtype=np.float32).copy()


def tile_boxes(size,tile_size=1024,overlap=128):
    width,height=size
    if min(width,height)<=0 or width*height>MAX_PIXELS:raise ValueError('Text-mask page exceeds pixel limit')
    if type(tile_size) is not int or tile_size not in (512,768,1024) or type(overlap) is not int or not 0<=overlap<tile_size//2:
        raise ValueError('Text-mask overlap must be below half the tile size')
    def starts(length):
        if length<=tile_size:return [0]
        return sorted(set(range(0,length-tile_size+1,tile_size-overlap))|{length-tile_size})
    xs,ys=starts(width),starts(height)
    if len(xs)*len(ys)>MAX_TILES:raise ValueError('Too many SAM-TS tiles; increase tile_size or reduce overlap')
    return [(x,y,min(x+tile_size,width),min(y+tile_size,height)) for y in ys for x in xs]


def segment_with(predict,image,mode='full_page',threshold=0.0,tile_size=1024,overlap=128,progress=None):
    """Small pure orchestration boundary makes coordinate/seam tests GPU-free."""
    width,height=image.size
    if min(width,height)<=0 or width*height>MAX_PIXELS:raise ValueError('Text-mask page exceeds pixel limit')
    if mode not in ('full_page','tiles') or not math.isfinite(threshold) or not -5<=threshold<=5:
        raise ValueError('Invalid SAM-TS inference options')
    boxes=[(0,0,width,height)] if mode=='full_page' else tile_boxes(image.size,tile_size,overlap)
    if len(boxes)==1:
        logits=predict(image)
        if logits.shape!=(height,width) or not np.isfinite(logits).all():raise ValueError('SAM-TS mask dimensions differ from source')
        if progress:progress(1,1)
        return (logits>threshold).astype(np.float32)
    accumulated=np.zeros((height,width),np.float32);weights=np.zeros_like(accumulated)
    for done,(left,top,right,bottom) in enumerate(boxes,1):
        crop=image.crop((left,top,right,bottom));logits=predict(crop)
        h,w=bottom-top,right-left
        if logits.shape!=(h,w) or not np.isfinite(logits).all():raise ValueError('Invalid SAM-TS tile mask')
        # Positive overlap ramps avoid unassigned edges and soften tile joins.
        def ramp(length):
            values=np.ones(length,np.float32);edge=min(overlap//2,length//2)
            if edge:
                fade=np.linspace(0.1,1,edge,dtype=np.float32)
                values[:edge]=fade;values[-edge:]=fade[::-1]
            return values
        weight=ramp(h)[:,None]*ramp(w)[None,:]
        accumulated[top:bottom,left:right]+=logits*weight
        weights[top:bottom,left:right]+=weight
        if progress:progress(done,len(boxes))
    return (accumulated/weights>threshold).astype(np.float32)


def build_model(config):
    import torch
    try:
        import einops
    except ModuleNotFoundError as exc:
        raise RuntimeError('Install einops in the existing ComfyUI venv; Hi-SAM training requirements are not needed') from exc
    from ._text_sam.image_encoder import ImageEncoderViT
    from ._text_sam.mask_decoder import MaskDecoder
    from ._text_sam.prompt_encoder import PromptEncoder
    from ._text_sam.modal_aligner import ModalAligner
    from ._text_sam.transformer import TwoWayTransformer
    from ._text_sam.hi_sam import HiSam
    if config.get('variant')!='vit_l_textseg' or config.get('image_size')!=1024 or config.get('hierarchical_detection') is not False:
        raise ValueError('Unsupported SAM-TS architecture')
    # Exactly the upstream vit_l SAM-TS architecture; no .pth checkpoint code.
    return HiSam(
        image_encoder=ImageEncoderViT(depth=24,embed_dim=1024,img_size=1024,mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),num_heads=16,patch_size=16,
            qkv_bias=True,use_rel_pos=True,global_attn_indexes=[5,11,17,23],window_size=14,out_chans=256),
        modal_aligner=ModalAligner(256,attn_layers=1,prompt_len=12),
        prompt_encoder=PromptEncoder(embed_dim=256,image_embedding_size=(64,64),input_image_size=(1024,1024),mask_in_chans=16),
        mask_decoder=MaskDecoder(num_multimask_outputs=3,transformer=TwoWayTransformer(depth=2,embedding_dim=256,mlp_dim=2048,num_heads=8),
            transformer_dim=256,iou_head_depth=3,iou_head_hidden_dim=256),
        pixel_mean=[123.675,116.28,103.53],pixel_std=[58.395,57.12,57.375])


class Backend:
    def __init__(self,folder,precision='bf16'):
        import torch
        from safetensors.torch import load_file
        if not torch.cuda.is_available():raise RuntimeError('SAM-TS requires CUDA')
        if precision not in ('bf16','fp32'):raise ValueError('Invalid SAM-TS precision')
        if precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('Select fp32 for GPUs without BF16 support')
        folder=validate(folder)
        from .model_bundle import identity
        self.model_name,self.revision=identity(folder,REPO,REV)
        self.model=build_model(json.loads((Path(folder)/'config.json').read_text()))
        state=load_file(str(Path(folder)/'model.safetensors'),device='cpu')
        self.model.load_state_dict(state,strict=True);del state
        self.model=self.model.eval().to('cuda')
        self.device='cuda';self.precision=precision
        started=time.monotonic()
        self._predict(Image.new('RGB',(96,64),'white'))
        torch.cuda.synchronize()
        self.warmup_complete=True;self.warmup_seconds=round(time.monotonic()-started,3)
        LOG.info('SAM-TS loaded: precision=%s, warmup=%.3fs',precision,self.warmup_seconds)

    def _predict(self,image):
        import torch
        array,size=prepare(image)
        tensor=torch.from_numpy(array).to(self.device)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.precision=='bf16'):
            output=self.model([{'image':tensor.contiguous(),'original_size':(IMAGE_SIZE,IMAGE_SIZE)}],multimask_output=False)
        logits=output[3][0,0].float().cpu().numpy()
        return restore(logits,size,image.size)

    def segment(self,image,**options):
        started=time.monotonic()
        result=segment_with(self._predict,image,**options)
        LOG.info('SAM-TS completed: page=%sx%s, mode=%s, foreground=%s, total=%.3fs',
                 image.width,image.height,options.get('mode','full_page'),int(result.sum()),time.monotonic()-started)
        return result
