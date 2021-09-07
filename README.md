# README

Partial analysis code for Zolin et al. Context-Dependent Representations of Movement in Drosophila Dopaminergic Reinforcement Pathways. *Nature Neuroscience* 2021.

This code implements several analyses on a dataset of dopaminergic signals impinging on the mushroom body in behaving fruit flies. To reproduce figure data for the figures listed below, clone this repository and download [dataset 2](https://www.nature.com/neuro/) into a directory called "data_" directly inside this repository. Note that these notebooks do not reproduce all figure data automatically. You will have to uncomment various flags in the notebooks to plot results for different mushroom body compartments (G2-5), behavioral output variables, or experiments. These flags should be fairly self-explanatory within the notebooks themselves. (Note: 'ASENSORY_AZ', 'CL_360_LOWFLOW_ACV', and 'CL_180_HighFlow_ACV' correspond to the asensory/sensory-restricted, low-flow, and high-flow experiments, respectively.)

Fig 2C: 1_asensory_state_class.ipynb

Fig 2D: 3_asensory_filters.ipynb

Fig 2E,F: 3_dan_from_behav_filters_trial_by_trial.ipynb

Fig 3D,E: 10A_odor_response_model_real_data_before_and_during.ipynb

Fig 4C: 10A_odor_response_model_real_data_before_and_during.ipynb

Fig 5B-E: 10A_odor_response_model_real_data_before_and_during.ipynb

Supp Fig 1: 1A_mvmt_initiation.ipynb

Ext Fig 4G: 4A_multi_filters.ipynb

Ext Fig 3G: 4_cl_filters.ipynb

Ext Fig 5G: 4_cl_filters.ipynb

Ext Fig 6: 10A_odor_response_model_real_data_before_and_during.ipynb

Ext Fig 7: 8_cl_odor_trig_corr_mats.ipynb

Ext Fig 8B-C: 9_cl_nested.ipynb

Ext Fig 9: 10A_odor_response_model_real_data_before_and_during.ipynb

The other notebooks implement additional analyses not shown in the publication.


All code uses Python 3.7 with Anaconda packages.
