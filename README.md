# Audio Quality Scoring Model

A machine learning model that predicts audio quality scores using Wav2Vec2 embeddings and XGBoost regression.

## Overview

This project implements an audio quality scoring system that:
- Extracts deep audio features using Facebook's Wav2Vec2 model
- Trains an XGBoost regressor to predict quality scores (0-5 scale)
- Handles audio files robustly with fallback loading mechanisms
- Achieves strong correlation between predicted and actual scores

## Model Architecture
## Dataset Access : `https://drive.google.com/drive/folders/16bmXNOnh1RvYVWBkwXAysy4dcoVpiYe9?usp=sharing`
### Feature Extraction
- **Model**: `facebook/wav2vec2-base-960h`
- **Method**: Mean-pooled embeddings from last hidden state
- **Output**: 768-dimensional feature vectors
- **Sample Rate**: 16kHz

### Regression Model
- **Algorithm**: XGBoost Regressor
- **Hyperparameters**:
  - n_estimators: 1500
  - max_depth: 7
  - learning_rate: 0.05
  - subsample: 0.9
  - colsample_bytree: 0.9
  - early_stopping_rounds: 50

## Results

### Performance Metrics
- **Training RMSE**: 0.0038
- **Training Pearson**: 0.9999
- **Validation RMSE**: 0.6151
- **Validation Pearson**: 0.6115

### Key Features
- 80/20 train-validation split
- StandardScaler normalization
- Early stopping to prevent overfitting
- Predictions clipped to [0, 5] range

## Project Structure
```
.
├── scoring-model.ipynb          # Main notebook
├── /mnt/data/
│   ├── embeddings/
│   │   ├── w2v2_train_embs.npy  # Cached training embeddings
│   │   ├── w2v2_test_embs.npy   # Cached test embeddings
│   │   └── *_files.npy          # Filename mappings
│   └── models/
│       └── xgb_w2v2.json        # Trained XGBoost model
└── submission.csv               # Final predictions
```

## Installation
```bash
pip install torch transformers soundfile librosa xgboost scikit-learn pandas numpy matplotlib plotly
```

## Usage

### Training
```python
# 1. Extract Wav2Vec2 embeddings
train_embs, train_files = get_embeddings_and_labels(
    train_df, TRAIN_WAV, TRAIN_EMB_PATH, TRAIN_FILES_PATH
)

# 2. Train XGBoost model
xgb_model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)])

# 3. Save model
xgb_model.save_model('xgb_w2v2.json')
```

### Inference
```python
# Load embeddings and scale
test_embs_scaled = scaler.transform(test_embs)

# Predict
test_preds = xgb_model.predict(test_embs_scaled)
predictions = np.clip(test_preds, 0.0, 5.0)
```

## Audio Loading

The model includes robust audio loading with fallback mechanisms:
```python
def robust_load(path, target_sr=16000, min_duration_seconds=1.0):
    try:
        wav, sr = sf.read(path)  # Primary: soundfile
    except:
        wav, sr = librosa.load(path)  # Fallback: librosa
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
```

## Dataset Format

### Input CSV
```csv
filename,label
audio_1.wav,3.5
audio_2.wav,4.0
```

### Output CSV
```csv
filename,label
audio_test_1.wav,3.2736
audio_test_2.wav,2.5420
```

## Hyperparameter Tuning

The model uses:
- **Validation-based early stopping**: Prevents overfitting
- **L2 regularization** (reg_lambda=1.0): Controls model complexity
- **Subsampling** (0.9): Improves generalization
- **Histogram-based tree method**: Faster training

## Known Limitations

1. **Browser Storage**: Does not use localStorage/sessionStorage (artifact constraint)
2. **Memory Constraints**: Large audio files processed in batches of 8
3. **Audio Format Support**: Requires WAV format or librosa-compatible formats
4. **Validation Performance**: Gap between training and validation suggests potential overfitting

## Future Improvements

- [ ] Implement K-fold cross-validation
- [ ] Experiment with ensemble methods
- [ ] Add data augmentation for audio files
- [ ] Fine-tune Wav2Vec2 on domain-specific data
- [ ] Implement attention-based pooling instead of mean pooling

## Dependencies

- torch >= 1.0
- transformers >= 4.0
- soundfile >= 0.10
- librosa >= 0.9
- xgboost >= 1.5
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.21

## License

This project is part of the SHL Intern Hiring Assessment 2025.

## Acknowledgments

- Wav2Vec2 model: Facebook AI Research
- XGBoost: Distributed (Deep) Machine Learning Community

## Citation
```bibtex
@misc{audio_quality_scorer_2025,
  title={Audio Quality Scoring with Wav2Vec2 and XGBoost},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/audio-quality-scorer}}
}
```
