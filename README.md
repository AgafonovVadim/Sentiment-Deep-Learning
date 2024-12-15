# Sentiment Analysis Deep Learning Model

## Description
This repository contains the implementation of a deep learning model designed to classify textual reviews into positive and negative categories. The project aims to provide an intuitive and efficient sentiment analysis solution that can be applied in various domains, such as customer feedback analysis, social media monitoring, and text analysis automation.

## Key Features
- **Dataset Preparation:** Collect and preprocess datasets suitable for sentiment analysis (e.g., IMDB, Amazon Reviews).
- **Label Creation:** For datasets without existing labels, generate labels using Large Language Models (LLMs) by providing a detailed task prompt.
- **Model Architecture:** Implements a text classification model using deep learning layers, such as `nn.Embedding`, `nn.LSTM`, and `nn.Linear`. More advanced architectures, including pre-trained models like BERT, are also supported.
- **Training and Evaluation:** Supports hyperparameter tuning, detailed metric calculation, and visualization of results.
- **Embeddings and Visualization:** Generates text embeddings, applies dimensionality reduction, and visualizes data with target labels in 2D space.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/agafonovvadim/sentiment-analysis-model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis-model
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare the dataset:**
   - Add your dataset files to the `data/` directory.
   - For datasets without labels, generate targets using an LLM:
     - Write a detailed prompt explaining the task (e.g., "Classify the sentiment of the following reviews as positive or negative...").
     - Use the LLM to annotate the dataset.
     - Save the labeled dataset in the `data/processed/` directory.

   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```

4. **Generate predictions:**
   ```bash
   python predict.py --text "Your text for sentiment analysis"
   ```

5. **Visualize embeddings (optional):**
   - Extract embeddings from the penultimate layer of the model.
   - Use dimensionality reduction (e.g., PCA, t-SNE) to project embeddings into 2D.
   - Plot the embeddings with colors representing the target labels.

## Future Enhancements
- Add support for multilingual sentiment analysis.
- Include real-time metrics visualization during training.
- Integrate with more advanced architectures like Transformer models.
- Develop an API for easy integration with external applications.

## License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contact
For questions or suggestions, feel free to reach out via email: [agafonovvadim@niuitmo.ru](mailto:your.email@example.com).
