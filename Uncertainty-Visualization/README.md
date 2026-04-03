
### Degrdation Processes 

Select the degradation process using the `--deg` flag in the configuration file:

* `lr` — applies spatial super‑resolution degradation
* `r_mask_fs` — applies imaging through opaque occluders 

#### Spatial Super-Resolution 

The `--radius` parameter (defined in `config.py` in pixels), sets the radius of the low‑pass filter in the Fourier domain. This determines the cut‑off frequency and therefore how much high‑frequency information is preserved in the degraded image.

This degradation process is implemented by the `circ` and `NA_cutoff` functions in `optical_utils.py`. 

#### Opaque Occluders 

The `--random_mask_size` parameter (defined in `config.py`, in pixels), sets the side length of a square occluding mask that is placed at random locations. 

This degradation process is implemented by `RandomSquareMasknFS` function in `utils.py`. 

### Warm-up 

The warm-up phase (see the [maunscript](https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=ol&uri=ol-51-8-2012&html=true) and Suppelmantry Note 2 for detials) is controlled by `--epcoh_initialize` flag in `config.py`. 

#### Related work

Hierarchical Uncertainty Exploration via Feedforward Posterior Trees, NeurIPS 2024. [[Project page]](https://eliasnehme.github.io/PosteriorTrees/) [[Paper]](https://neurips.cc/virtual/2024/poster/94955)

## For more details, see the [README file](https://github.com/matankleiner/All-Optical-SR/blob/main/README.md) in the main direcorty. 
