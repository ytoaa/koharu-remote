"""Estimate a smooth background offset from unmasked, similar-tone pixels."""
import numpy as np
from scipy import ndimage

MAX_PIXELS = 16 * 1024 * 1024
WORK_SIDE = 512


def local_color_match(target, reference, exclude_mask, strength=1.0, radius=32,
                      max_shift=0.10, sample_tolerance=0.15, mode='luminance'):
    """Pure array implementation; neither inputs nor the compositing mask change.

    Similar-luminance bands prevent a white margin dominating a gray panel.
    Normalized convolution excludes masked samples rather than replacing them
    with zeros. Regions without nearby samples inherit the nearest supported
    offset in that tone band. This cannot reconstruct missing background art.
    """
    target=np.asarray(target,dtype=np.float32)
    reference=np.asarray(reference,dtype=np.float32)
    exclude_mask=np.asarray(exclude_mask,dtype=np.float32)
    if target.ndim!=3 or target.shape[-1]!=3 or reference.shape!=target.shape or exclude_mask.shape!=target.shape[:2]:
        raise ValueError('Align target, reference and mask dimensions with standard graph nodes')
    height,width=target.shape[:2]
    if min(height,width)<=0 or height*width>MAX_PIXELS:
        raise ValueError('Local color match image exceeds pixel limit')
    if not all(np.isfinite(a).all() for a in (target,reference,exclude_mask)):
        raise ValueError('Expected finite images and mask')
    if (not np.isfinite([strength,max_shift,sample_tolerance]).all() or
        not 0<=strength<=1 or not 0<=max_shift<=0.5 or not 0<sample_tolerance<=1 or
        type(radius) is not int or not 1<=radius<=256 or mode not in ('luminance','rgb')):
        raise ValueError('Invalid local color match options')
    if not strength or not max_shift or np.all(exclude_mask>=0.5):
        return target
    scale=min(1.0,WORK_SIDE/max(height,width))
    step=int(np.ceil(1/scale))
    # Mask pooling excludes the entire sampling cell, avoiding text leakage
    # when a narrow text stroke falls between samples of a larger source.
    excluded=ndimage.maximum_filter(exclude_mask>=0.5,size=step,mode='nearest')[::step,::step]
    src=target[::step,::step]
    ref=reference[::step,::step]
    delta=ref-src
    valid=(~excluded)&(np.max(np.abs(delta),axis=-1)<=sample_tolerance)
    if np.count_nonzero(valid)<8:
        return target
    coefficients=np.array([0.2126,0.7152,0.0722],dtype=np.float32)
    luminance=src@coefficients
    differences=(delta@coefficients)[...,None] if mode=='luminance' else delta
    channels=differences.shape[-1]
    accumulated=np.zeros((*src.shape[:2],channels),np.float32)
    coverage=np.zeros(src.shape[:2],np.float32)
    sigma=max(0.5,radius/(3*step))
    for center in np.linspace(0,1,9):
        membership=np.maximum(0,1-np.abs(luminance-center)/0.15)
        samples=membership*valid
        if samples.sum()<2:
            continue
        support=ndimage.gaussian_filter(samples,sigma,mode='nearest',truncate=3)
        supported=support>0.01
        if not supported.any():
            continue
        nearest=None
        if not supported.all():
            nearest=ndimage.distance_transform_edt(~supported,return_distances=False,return_indices=True)
        field=np.empty((*src.shape[:2],channels),np.float32)
        for channel in range(channels):
            numerator=ndimage.gaussian_filter(differences[...,channel]*samples,sigma,mode='nearest',truncate=3)
            values=np.divide(numerator,support,out=np.zeros_like(numerator),where=supported)
            if nearest is not None:
                values[~supported]=values[tuple(nearest[:,~supported])]
                # Smooth the joins between nearest-supported areas inside a
                # large exclusion region; do not introduce Voronoi seams.
                values=ndimage.gaussian_filter(values,sigma,mode='nearest',truncate=3)
            field[...,channel]=values
        accumulated+=field*membership[...,None]
        coverage+=membership
    field=np.divide(accumulated,coverage[...,None],out=np.zeros_like(accumulated),where=coverage[...,None]>0)
    # Coordinates use the same pixel grid as the sampling stride.
    if step>1:
        coordinates=np.meshgrid(np.arange(height,dtype=np.float32)/step,np.arange(width,dtype=np.float32)/step,indexing='ij')
        field=np.stack([ndimage.map_coordinates(field[...,c],coordinates,order=1,mode='nearest',prefilter=False) for c in range(channels)],axis=-1)
    correction=np.clip(field,-max_shift,max_shift)*strength
    return np.clip(target+correction,0,1).astype(np.float32)


class KoharuLocalColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'target':('IMAGE',),'reference':('IMAGE',),'exclude_mask':('MASK',),
            'strength':('FLOAT',{'default':1.0,'min':0.,'max':1.,'step':0.05}),
            'radius':('INT',{'default':32,'min':1,'max':256}),
            'max_shift':('FLOAT',{'default':0.10,'min':0.,'max':0.5,'step':0.01}),
            'sample_tolerance':('FLOAT',{'default':0.15,'min':0.01,'max':1.,'step':0.01}),
            'mode':(['luminance','rgb'],)}}
    RETURN_TYPES=('IMAGE',)
    RETURN_NAMES=('corrected',)
    FUNCTION='match'
    CATEGORY='Koharu/Image'
    def match(self,target,reference,exclude_mask,**options):
        import torch
        if target.shape[0]!=1 or reference.shape[0]!=1 or exclude_mask.shape[0]!=1:
            raise ValueError('Local color match expects one image and mask')
        source=target.detach().cpu().numpy()[0]
        result=local_color_match(source,reference.detach().cpu().numpy()[0],exclude_mask.detach().cpu().numpy()[0],**options)
        return (torch.from_numpy(result[None]).to(device=target.device,dtype=target.dtype),)


NODE_CLASS_MAPPINGS={'KoharuLocalColorMatch':KoharuLocalColorMatch}
