# The feature landscape of Mouse Visual Cortex
Analysis code and ANN models for LINK-TO-PAPER.

## Analysis scripts (analysis/)

- analyse_variance_validation.py: Analyse the response variance of neurons to own- vs other-stimuli (Figure 6)
- analyse_wdf.py: Analyse population activity to own- vs other-stimuli (Figure 6)
- apply_mask.py: Apply the spatial mask to preferred images.
- calculate_embedding_hierarchy.py: Calculate the local overlap between manifolds using kNN (Figure 3)
- calculate_FFT_statistics.py: Calculate radial and axial average of FFT spectra (Figure 5)
- calculate_image_statistics.py: Calculate low-level image statistics (Figure 5)
- calculate_noise_ceiling.py: Estimate explainable variance (Figure 1)
- calculate_linear_predictability.py: Fit PLS regression model to predict neuronal responses (Figure 1)
- calculate_spatial_mask.py: Fit 2D Gaussian to spatial mask layer.
- combine_data.py: Combines processed data from individual recordings into a single file for ANN training.
- compute_distance_correlation.py: Compute the distance and partial distance correlation between pairs of regions (Figure 2)
- evaluate_optstim.py: Remove preferred images of badly fit neurons.
- extract_activations.py: Present stimuli to ANN models and save activation matrix.
- get_image_embedding.py: Given a SimCNE model, generate the image embedding and UMAP embedding (Figure 3,4)
- get_model_performance.py: Assess performance of ANN model (Figure 1).
- get_representative_images.py: Sort images by representativeness (Figure 5)
- models.py: Collection of pytorch ANN models.
- plot_params.py: Parameters for generating Figures.
- segment_retmap.py: Perform retinotopic mapping (Figure 6).
- test_generalisation_insilico.py: Present preferred images back to ANNs and measure the activity (Figure 6)
- train_SimCNE.py: Train a SimCNE model on preferred images.
- utils.py: Various functions shared across scripts.
- Figurex.ipynb: Jupyter notebooks to generate figures used in the paper.

## ANN model scripts (train/)

- calculate_noise_ceiling.py (obsolete): Calculate the noise ceiling to include only reliable neurons. Required for training ANN but thresholding is done separately now.
- dataloader.py: Dataloader class for training ANN model.
- make_optStim.py: Run Lucent preferred image generation.
- optimalStimuli.py: Run Lucent preferred image generation.
- model.py: Collection of pytoch ANN models.
- save_full_model_rt.py: Runs ANN training.
- shallow_simclr_backbone.py: Shallow network architecture used for ANN.
- train_utils.py: Various functions for ANN training.
- rudi_utils.py: More functions for ANN training.

## Data accessibility

Processed data used in this study can be accessed at []. For raw data, please contact stuart.trenholm@mcgill.ca

## Visual feature atlas explorer

A [Google Colab](https://colab.research.google.com/drive/1DOt-KBKAFenmBIaDJ763ARsAvo8HDFzc?usp=sharing) is available to explore the preferred images generated in this study.

## Contact

For questions regarding the study, please email stuart.trenholm@mcgill.ca or rudi.tong@mcgill.ca
For specific questions regarding ANN model, please email blake.richards@mila.quebec

![](https://raw.githubusercontent.com/ruditong/MouseFeatureLandscape/main/images/QR_trenholmlab.png?token=GHSAT0AAAAAACIKL5AEKY57YKVLR7Y64CEQZI4FQVQ)