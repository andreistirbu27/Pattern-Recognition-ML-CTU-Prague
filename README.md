# Pattern Recognition & Machine Learning (CTU Prague) - RPZ assignments

This repo contains my work for the CTU Prague **B4B33RPZ / RPZ** course assignments (Python templates + my solutions and experiments).
Most assignments have an accompanying notebook (`*.ipynb`) plus a reference implementation in `*.py`.

## Repo layout

- `assignment_basics/` - NumPy warm-up + simple data exploration/visualisation
- `assignment_bayes/` - Bayes decision theory on a 2-class Gaussian setup
- `assignment_perceptron/` - Perceptron, feature lifting, and classification
- `assignment_logreg/` - Logistic regression with IRLS/Newton style optimisation
- `assignment_svm/` - SVM basics + kernels + prediction utilities
- `assignment_kmeans/` - K-means + k-means++ + applications (incl. color quantization)
- `assignment_adaboost/` - AdaBoost with decision stumps
- `assignment_parzen/` - Parzen window density estimation + Bayes classification
- `assignment_mle_map_bayes/` - MLE/MAP parameter estimation + Bayesian classifier
- `assignment_minimax/` - minimax / robust decision rules for classification
- `assignment_cnn/` - CNN assignment: (1) NN “by hand” in NumPy, (2) CNNs in PyTorch, plus finetuning

## What I implemented (per-lab summary)

### `assignment_basics`
- Mean image computation and visualisation (`find_mean`, `show_mean`)
- Covariance computation + visualisation (`covariance_matrix`, `compute_measurement`, `show_cov`)
- Simple classification and reporting utilities (`classify_image`, `classification_error`, `compute_letter_mean`, `print_classification_error`)
- Helper visualisations like montage creation (`create_montage`)

### `assignment_bayes`
- Bayes classifier for Gaussians (`bayes_classifier`)
- 0-1 and general cost classification errors (`classification_error_01`, `classification_error`)
- Empirical risk / expected risk utilities (`compute_risk`, `find_strategy_2normal`, `risk_fix_q`, `worst_risk_cont`, `minimax_strategy_cont`)
- Plotting helpers for decision boundaries/risks (`plot_boundary`, `plot_risk`)

### `assignment_perceptron`
- Perceptron training (`perceptron`)
- Feature lifting (incl. quadratic) (`lift_dimension`, `lift_dimension_quadratic`)
- Classification and error computation (`classif_perc`, `compute_test_error`)
- Data visualisation helpers (`plot_line`, `pplot`, `show_classification`)

### `assignment_logreg`
- Logistic loss + gradient + Hessian (`logistic_loss`, `logistic_loss_gradient`, `logistic_loss_Hessian`)
- Regularised training with Newton / IRLS style updates (`train_logistic_regression`)
- Prediction + boundary plotting + reporting (`classify_images`, `plot_boundary`, `classification_error`)
- Feature lifting helpers (`lift_dimension`, `lift_dimension_quadratic`)

### `assignment_svm`
- Kernel computation (`kernel_matrix`)
- SVM training wrapper (`my_svm`)
- Prediction utilities (`svm_predict`, `classify_images`)
- Visualisation/reporting helpers (`plot_boundary`, `show_classification`, `compute_measurement_lr_cont`)

### `assignment_kmeans`
- K-means (Lloyd’s algorithm) (`k_means`)
- K-means++ initialisation (`k_meanspp`)
- Multiple restarts / best solution (`k_means_multiple_trials`)
- Pixel clustering for color quantization (`quantize_colors`)
- Visualisation helpers (`show_clusters`)

### `assignment_adaboost`
- AdaBoost training loop (`adaboost`)
- Weak learner search (decision stump) (`find_best_weak`, `classify_weak`, `compute_error`)
- Strong classifier prediction (`classify_strong`)
- Plotting utilities (`plot_boundary`, `show_classification`, `show_classification_slow`, `plot_graph`)

### `assignment_parzen`
- Parzen density estimation (`parzen`)
- Bayesian classification using estimated densities (`bayes_classifier_parzen`)
- Classification error + plotting (`classification_error`, `plot_boundary`, `plot_density`)

### `assignment_mle_map_bayes`
- MLE and MAP estimation for distributions used in the assignment (`ml_estim_*`, `map_estim_*`)
- Bayesian classification based on estimated parameters (`bayesian_classifier`)
- Error + risk utilities (`classification_error`, `compute_measurement_lr_cont`)
- Visualisation helpers (`plot_boundary`, `plot_distribution`)

### `assignment_minimax`
- Continuous minimax strategy computation (`minimax_strategy_cont`)
- Worst-case risk evaluation (`worst_risk_cont`)
- Supporting measurement / plotting helpers (`compute_measurement_lr_cont`, `plot_risk`)

### `assignment_cnn`
**Part 1 (NumPy):** small neural-net framework with forward/backward passes
- Layers: `Linear`, `ReLU`, `Sigmoid`
- Loss: `SE` (squared error)
- Training utilities: `SGD`, plus plotting/evaluation helpers

**Part 2 (PyTorch):**
- Baselines: `FCNet`, `SimpleCNN`
- Stronger CNN: `MyNet` (Conv + BN + ReLU + pooling/dropout + classifier head)
- Transfer learning scaffold: `FinetuneNet` (load torchvision backbone, replace head, freeze/unfreeze, save checkpoint)

## Running the notebooks

Most work is in Jupyter notebooks inside each assignment folder.

Typical setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you keep one, otherwise install numpy/matplotlib/scipy/sklearn/torch/torchvision
jupyter lab
