# All Optical Uncertainty Visualization

### Official pytorch implementation of the paper "All-optical uncertainty visualization for ill-posed image restoration tasks" [[Paper]](https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=ol&uri=ol-51-8-2012)

### For system requiements and usage instruction see the [README file](https://github.com/matankleiner/All-Optical-SR/blob/main/README.md) in the main direcorty. 

## Degrdation Processes 

Select the degradation process using the `--deg` flag in the configuration file:

* `lr` — applies spatial super‑resolution degradation
* `r_mask_fs` — applies imaging through opaque occluders 

### Spatial Super-Resolution 

The `--radius` parameter (defined in `config.py` in pixels), sets the radius of the low‑pass filter in the Fourier domain. This determines the cut‑off frequency and therefore how much high‑frequency information is preserved in the degraded image.

This degradation process is implemented by the `circ` and `NA_cutoff` functions in `optical_utils.py`. 

### Opaque Occluders 

The `--random_mask_size` parameter (defined in `config.py`, in pixels), sets the side length of a square occluding mask that is placed at random locations. 

This degradation process is implemented by `RandomSquareMasknFS` function in `utils.py`. 

## Warm-up 

The warm-up phase (see the [maunscript](https://opg.optica.org/viewmedia.cfm?r=1&rwjcode=ol&uri=ol-51-8-2012&html=true) and Suppelmantry Note 2 for detials) is controlled by `--epcoh_initialize` flag in `config.py`. 

## Related work

Elias Nehme, Rotem Mulayoff and Tomer Michaeli, "Hierarchical Uncertainty Exploration via Feedforward Posterior Trees", NeurIPS 2024. [[Project page]](https://eliasnehme.github.io/PosteriorTrees/) [[Paper]](https://neurips.cc/virtual/2024/poster/94955)

Elias Nehme and Tomer Michaeli, "Generative AI for Solving Inverse Problems in Computational Imaging", XRDS: Crossroads, The ACM Magazine for Students 2025. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3703401)

### Citation 

If you use this code for your research, please cite our paper:

```
@article{kleiner2025can,
  title={Can the success of digital super-resolution networks be transferred to passive all-optical systems?},
  author={Kleiner, Matan and Michaeli, Lior and Michaeli, Tomer},
  journal={Nanophotonics},
  volume={14},
  number={19},
  pages={3181--3190},
  year={2025},
  publisher={De Gruyter}
}
```

```
@article{kleiner2026all,
  author = {Kleiner, Matan and Michaeli, Tomer},
  journal = {Opt. Lett.},
  number = {8},
  pages = {2012--2015},
  publisher = {Optica Publishing Group},
  title = {All-optical uncertainty visualization for ill-posed image restoration tasks},
  volume = {51},
  month = {Apr},
  year = {2026},
}
```
