# CNN Image Classification with Fine-Tuning

This repository contains code for training a Convolutional Neural Network (CNN) for image classification using PyTorch. The project demonstrates fine-tuning models on different datasets, comparing model performance, and implementing Grad-CAM for visualizing model predictions.

## 1. Setup
To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    ```
   
2. Navigate to the project folder:
    ```bash
    cd your-repo
    ```

## 2. Training AlexNet on CIFAR-10
You can train the AlexNet model on the CIFAR-10 dataset by running the following notebook:

- Run the notebook `mm23abp.ipynb` to train AlexNet on the CIFAR-10 dataset.

Training progress and results will be saved in a CSV file.

## 3. Fine-Tuning Model with Frozen Layers
The project also demonstrates fine-tuning with a pre-trained model while freezing the base convolutional layers. Follow these steps:

1. Load the pre-trained AlexNet model.
2. Freeze the convolutional layers by modifying the code like this:
    ```python
    for param in model.features.parameters():
        param.requires_grad = False
    ```
3. Modify the last layer to suit CIFAR-10 classification:
    ```python
    model.classifier[6] = nn.Linear(4096, 10)  # CIFAR-10 has 10 classes
    ```
4. Train the model with the modified classifier layers. This will fine-tune the model while keeping the pre-trained convolutional layers intact.

## 4. Model Comparisons
Once the models have been trained, you can compare the performance of the original AlexNet on CIFAR-10 with the fine-tuned version.

- Generate and display graphs of training and validation accuracy and loss for both models. Example code to plot accuracy/loss graphs:

    ```python
    import matplotlib.pyplot as plt
    
    # Example: Plot Training and Validation Accuracy
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.show()
    ```

## 5. Training AlexNet on TinyImageNet30
To train AlexNet on the TinyImageNet30 dataset, follow these steps:

1. Run the script `train_alexnet_tiny.py` to train the model:
    ```bash
    python train_alexnet_tiny.py
    ```

2. Similar to the CIFAR-10 training process, the results will be saved in a CSV file for analysis.

## 6. Comparing Results on TinyImageNet30
- Compare the performance of the model trained on TinyImageNet30 with the one trained on CIFAR-10.
- Display graphs of training and validation accuracy and loss for both models, as demonstrated in Section 4.

## 7. Interpretation of Results

### 7.1 Grad-CAM Visualization
Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize which parts of an image the model focuses on while making predictions.

1. Install `torchcam` to enable Grad-CAM:
    ```bash
    pip install torchcam
    ```

2. Run `grad_cam_visualization.py` to apply Grad-CAM on both correctly and incorrectly classified images:
    ```bash
    python grad_cam_visualization.py
    ```

3. The output will display images with overlaid heatmaps that highlight the areas the model used to make its predictions.

### 7.2 Comments on Model Predictions
Analyze the Grad-CAM visualizations to understand why the model made certain predictions, and evaluate the reasons for correct and incorrect classifications.

### 7.3 Improvement Strategies
To further improve model performance, consider the following techniques:
- Implement data augmentation to improve generalization.
- Modify the model architecture for better feature extraction.
- Apply fine-tuning to more layers of the network.
- Use techniques like dropout to reduce overfitting.
- Tune hyperparameters (learning rate, batch size, etc.) for better optimization.

## Conclusion
This repository demonstrates CNN-based image classification, fine-tuning, and result interpretation through Grad-CAM visualizations. The provided scripts and methods can be adapted to your specific use case or extended for different datasets and models.
