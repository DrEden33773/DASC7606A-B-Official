# Deep Learning Image Classification Assignment Repository (DASC7606A-B)

This project guides you through the complete life-cycle of a deep learning project for image classification in computer vision. You'll build, train, and evaluate convolutional neural networks (CNNs) using the CIFAR-10 dataset.

## 🎯 Learning Objectives

By completing this project, you will:

- **Understand the complete deep learning workflow** from data collection to model evaluation
- **Master data preprocessing techniques** including cleaning, normalization, and augmentation
- **Build and train CNN architectures** from scratch and using transfer learning
- **Implement training best practices** including optimization, regularization, and early stopping
- **Evaluate model performance** using comprehensive metrics and visualizations
- **Develop debugging skills** for identifying and fixing common deep learning issues

## 📚 Course Structure

The project is divided into 6 sequential stages, each building upon the previous:

### Stage 1: Data Collection (`01_data_collection.ipynb`)

- **Key Concepts**: Dataset downloading, exploration, visualization, statistical analysis
- **Learning Goals**: Understand dataset structure, class distribution, and compute normalization statistics
- **Deliverable**: Downloaded CIFAR-10 dataset with saved statistics

### Stage 2: Data Augmentation (`02_data_augmentation.ipynb`)

- **Key Concepts**: Image augmentation techniques, data balancing, transformation pipelines
- **Learning Goals**: Apply various augmentation methods to improve model generalization
- **Deliverable**: Augmented dataset with increased diversity

### Stage 3: Model Building (`03_model_building.ipynb`)

- **Key Concepts**: CNN architectures, layer design, transfer learning, model complexity
- **Learning Goals**: Design and implement different CNN architectures
- **Deliverable**: A simple CNN model ready for training

### Stage 4: Model Training (`04_model_training.ipynb`)

- **Key Concepts**: Training loops, optimization algorithms, regularization, hyperparameter tuning
- **Learning Goals**: Train models effectively while avoiding overfitting
- **Deliverable**: Trained models with training history

### Stage 5: Model Evaluation (`05_model_evaluation.ipynb`)

- **Key Concepts**: Performance metrics, confusion matrices, ROC curves, error analysis
- **Learning Goals**: Comprehensively evaluate model performance and identify improvement areas
- **Deliverable**: Complete evaluation report with visualizations

## 🛠️ Setup Instructions

### Prerequisites

- **Python**: 3.13 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 2GB free space for datasets and models
- **GPU**: CUDA-compatible GPU (optional, but recommended for faster training)

### Installation

1. **Clone the repository**:

```bash
git clone git@github.com:hkukend/DASC7606A-B.git
cd DASC7606A-B
```

2. **Install dependencies** using uv (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

3. **Verify installation**:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
jupyter notebook --version
```

## 🚀 Quick Start

1. **Launch Jupyter Notebook**:

```bash
jupyter notebook
```

2. **Run notebooks in order**:
   - Start with `01_data_collection.ipynb`
   - Complete each notebook before moving to the next
   - Follow the checkpoint questions to test your understanding

3. **Monitor progress**:
   - Check `results/` directory for evaluation outputs
   - Check `models/` directory for saved models
   - Check `data/` directory for processed datasets

## 📊 Project Structure

```
DASC7606A-B/
├── 01_data_collection.ipynb          # Data downloading and exploration
├── 02_data_augmentation.ipynb        # Data augmentation techniques
├── 03_model_building.ipynb           # CNN architecture design
├── 04_model_training.ipynb           # Model training and optimization
├── 05_model_evaluation.ipynb         # Performance evaluation
├── scripts/                          # Reusable utility functions
│   ├── data_download.py              # Dataset downloading utilities
│   ├── data_augmentation.py          # Augmentation utilities
│   ├── model_architectures.py        # CNN model definitions
│   ├── train_utils.py                # Training helper functions
│   └── evaluation_metrics.py         # Evaluation and visualization
├── data/                             # Dataset storage (Will be generated during runtime)
│   ├── raw/                          # Original downloaded data
│   ├── processed/                    # Cleaned and processed data
│   └── augmented/                    # Augmented training data
├── models/                           # Saved model checkpoints (Will be generated during runtime)
├── results/                          # Evaluation outputs and visualizations (Will be generated during runtime)
├── main.py                           # The main script to run the whole pipeline
├── pyproject.toml                    # Project dependencies
└── uv.lock                          # Dependency lock file
```

## 🧪 Assignment: CIFAR-100 Extension

- **Goal**: After you familiarize yourself with the end-to-end CIFAR-10 pipeline (scripts and notebooks), extend this repository to train a better-performing model on **CIFAR-100**.
- **Data availability**: A CIFAR-100 download function is already provided in `scripts/data_download.py`.
- **What to improve**:
  - **Augmentation**: Enhance `scripts/data_augmentation.py` to better regularize and diversify training data.
  - **Model**: Design a more effective CNN (Even the SOTA architectures!) in `scripts/model_architectures.py` suitable for 100 classes.
  - **Hyperparameters**: Tune training hyperparameters passed to `main.py` (e.g., learning rate, weight decay, batch size, scheduler, epochs).
- **Scope of changes (must follow)**:
  - Modify only: `scripts/data_augmentation.py`, `scripts/model_architectures.py`, `scripts/train_utils.py`, and the hyperparameter arguments you feed into `main.py`.
  - Do not change the notebooks; use them to visualize, experiment, and understand the pipeline.
- **How to run CIFAR-100 training (example)**:

  ```bash
  python main.py --dataset cifar100 --batch_size 128 --epochs 50 --lr 0.001 --weight_decay 1e-4
  ```

## 🌐 Submission: Packaging Scripts

**If your student ID is 30300xxxxx, organize your submission as follows:**

```text
30300xxxxx.zip
├── scripts/                          # Reusable utility functions
│   ├── data_download.py              # Dataset downloading utilities
│   ├── data_augmentation.py          # Augmentation utilities
│   ├── model_architectures.py        # CNN model definitions
│   ├── train_utils.py                # Training helper functions
│   └── evaluation_metrics.py         # Evaluation and visualization
└── main.py                           # The main script to run the whole pipeline
```

- **Code Files:** All your modified code files.
- **Submission Format:** Zip archive with your student ID as the filename.

### Submission Deadline

**Deadline:** Oct. 26 (23:59 GMT +8), 2025

**Late Submission Policy:**

- 10% penalty within 1 day late.
- 20% penalty within 2 days late.
- 50% penalty within 7 days late.
- 100% penalty after 7 days late.

## 📈 Grading Criteria

Your submission will be evaluated based on criterion:

### Model Performance

We will re-run your `main.py` script to evaluate your model's performance on the test set. Please set your best hyperparameters as default in main.py file.

**Important Considerations:**

1. **Error-Free Execution:** Your code must run without any errors, including CUDA OOM error under HKU GPU Farm environment.
2. **Correct Training and Evaluation:** Ensure your model is trained and evaluated correctly according to the instructions. Training on testset is strictly prohibited.
3. **Reasonable Performance:** Your model should achieve a reasonable macro avg. F1-score on the test set.
4. **Execution Time:** The execution time should be less than 12 hours.

**Grading Breakdown (based on macro avg. F1-score on the test set):**

- **F1-score >= 0.85:** Full marks (100%)
- **F1-score >= 0.80:** 90% of the marks
- **F1-score >= 0.75:** 80% of the marks
- **F1-score >= 0.70:** 70% of the marks
- **F1-score >= 0.65:** 60% of the marks
- **F1-score >= 0.55:** 50% of the marks
- **F1-score < 0.55/Fail to reproduce/Overtime:** No marks (0%)

## ⚙️ Configuration

Key hyperparameters can be modified in the respective notebooks and scripts:

- **Batch Size**: 32-128 (default: 64)
- **Learning Rate**: 0.0001-0.01 (default: 0.001)
- **Epochs**: 10-100 (default: 50)
- **Model Architecture**: A SimpleCNN
- **Augmentation**: Rotation, flip, color jitter parameters

## 🔧 Key Technologies

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities and pre-trained models
- **Albumentations**: Advanced image augmentation
- **Scikit-learn**: Evaluation metrics and utilities
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## 🎓 Pedagogical Features

### Checkpoint Questions

Each notebook includes checkpoint questions to reinforce learning:

- **Concept Checks**: Test understanding of key concepts
- **Code Challenges**: Apply concepts through coding exercises
- **Debugging Scenarios**: Identify and fix common issues

### Common Pitfalls

Look for these warning boxes throughout the notebooks:
> ⚠️ **Common Pitfall**: Overfitting on training data
>
> Monitor validation loss and use early stopping to prevent overfitting.

### Sample Outputs

Each stage provides expected outputs to help validate your progress:

- Sample visualizations
- Expected accuracy ranges
- Performance benchmarks

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**:
   - Reduce batch size
   - Use gradient accumulation
   - Close unused applications

2. **Slow Training**:
   - Enable GPU acceleration
   - Reduce image resolution
   - Use smaller model architectures

3. **Poor Performance**:
   - Check data preprocessing
   - Adjust learning rate
   - Increase model complexity
   - Add more augmentation

4. **Installation Issues**:
   - Ensure Python 3.13+
   - Use virtual environment
   - Check CUDA compatibility for GPU support

### Getting Help

1. **Check the logs**: Look for error messages in notebook outputs
2. **Review documentation**: Read function docstrings and comments
3. **Validate outputs**: Compare your results with sample outputs
4. **Debug systematically**: Use print statements and breakpoints
