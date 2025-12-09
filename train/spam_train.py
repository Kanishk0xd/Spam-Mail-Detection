import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

class SpamTrainerLightGBM:
    def __init__(self, data_path='../data/combined_multi_spam_dataset.csv'):
        """
        Initialize spam trainer with LightGBM
        
        Args:
            data_path: Path to the combined dataset CSV
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tfidf = None
        self.model = None
        self.results = {}
        
    def load_data(self):
        """Load and validate dataset"""
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)
        
        if not os.path.exists(self.data_path):
            print(f" Error: Dataset not found at {self.data_path}")
            print("\nPlease run prepare_dataset.py first to create the dataset:")
            print("   python prepare_dataset.py")
            return False
        
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"   Total emails: {len(self.df)}")
        print(f"   Spam emails: {sum(self.df['label'] == 1)} ({sum(self.df['label'] == 1)/len(self.df)*100:.1f}%)")
        print(f"   Ham emails: {sum(self.df['label'] == 0)} ({sum(self.df['label'] == 0)/len(self.df)*100:.1f}%)")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print(f"\n Warning: Found {missing.sum()} missing values")
            self.df = self.df.dropna()
            print(f"   Dropped missing values. New size: {len(self.df)}")
        
        # Basic text statistics
        self.df['text_length'] = self.df['text'].str.len()
        print(f"\n Text Length Statistics:")
        print(f"   Average: {self.df['text_length'].mean():.0f} characters")
        print(f"   Median: {self.df['text_length'].median():.0f} characters")
        print(f"   Min: {self.df['text_length'].min()}")
        print(f"   Max: {self.df['text_length'].max()}")
        
        return True
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['text'], 
            self.df['label'], 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['label']
        )
        
        print(f"Training set: {len(self.X_train)} emails")
        print(f"  - Spam: {sum(self.y_train == 1)}")
        print(f"  - Ham: {sum(self.y_train == 0)}")
        print(f"\nTest set: {len(self.X_test)} emails")
        print(f"  - Spam: {sum(self.y_test == 1)}")
        print(f"  - Ham: {sum(self.y_test == 0)}")
    
    def create_tfidf_features(self):
        """
        Create TF-IDF features with optimized parameters
        """
        print("\n" + "="*70)
        print("CREATING TF-IDF FEATURES")
        print("="*70)
        
        # TF-IDF Vectorizer with optimal parameters for spam detection
        self.tfidf = TfidfVectorizer(
            max_features=10000,           # Use top 10k features
            ngram_range=(1, 3),            # Unigrams, bigrams, and trigrams
            stop_words='english',          # Remove English stop words
            min_df=3,                      # Ignore terms that appear in fewer than 3 documents
            max_df=0.9,                    # Ignore terms that appear in more than 90% of documents
            sublinear_tf=True,             # Use sublinear tf scaling (1 + log(tf))
            use_idf=True,                  # Enable IDF
            smooth_idf=True,               # Add 1 to document frequencies
            norm='l2',                     # L2 normalization
            strip_accents='unicode',       # Remove accents
            lowercase=True,                # Convert to lowercase
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens (min 2 chars)
        )
        
        print("TF-IDF parameters:")
        print(f"  - Max features: 10,000")
        print(f"  - N-gram range: (1, 3) - unigrams, bigrams, trigrams")
        print(f"  - Min document frequency: 3")
        print(f"  - Max document frequency: 0.9")
        print(f"  - Sublinear TF: Yes")
        print(f"  - IDF: Yes (smoothed)")
        print(f"  - Normalization: L2")
        
        print("\nTransforming training data...")
        X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        
        print(f"✓ TF-IDF matrix created:")
        print(f"  - Shape: {X_train_tfidf.shape}")
        print(f"  - Features: {X_train_tfidf.shape[1]}")
        print(f"  - Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")
        
        print("\nTransforming test data...")
        X_test_tfidf = self.tfidf.transform(self.X_test)
        print(f"✓ Test data transformed: {X_test_tfidf.shape}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_lightgbm(self, X_train_tfidf, X_test_tfidf, tune_hyperparameters=False):
        """
        Train LightGBM model
        
        Args:
            X_train_tfidf: TF-IDF transformed training data
            X_test_tfidf: TF-IDF transformed test data
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70)
        
        if tune_hyperparameters:
            print("Tuning hyperparameters with cross-validation...")
            
            # Hyperparameter tuning
            best_score = 0
            best_params = None
            
            param_grid = {
                'num_leaves': [31, 50, 70],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [100, 200],
                'max_depth': [-1, 10, 20]
            }
            
            print("Testing parameter combinations...")
            for num_leaves in param_grid['num_leaves']:
                for lr in param_grid['learning_rate']:
                    for n_est in param_grid['n_estimators']:
                        for depth in param_grid['max_depth']:
                            params = {
                                'objective': 'binary',
                                'metric': 'binary_logloss',
                                'num_leaves': num_leaves,
                                'learning_rate': lr,
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'feature_fraction': 0.8,
                                'bagging_fraction': 0.8,
                                'bagging_freq': 5,
                                'verbose': -1,
                                'random_state': 42
                            }
                            
                            model = lgb.LGBMClassifier(**params)
                            scores = cross_val_score(model, X_train_tfidf, self.y_train, 
                                                    cv=3, scoring='f1')
                            score = scores.mean()
                            
                            if score > best_score:
                                best_score = score
                                best_params = params
                            
                            print(f"  Params: leaves={num_leaves}, lr={lr}, n={n_est}, depth={depth} -> F1={score:.4f}")
            
            print(f"\n✓ Best parameters found: {best_params}")
            print(f"  Best CV F1 score: {best_score:.4f}")
            self.model = lgb.LGBMClassifier(**best_params)
        else:
            # Use optimized default parameters
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 50,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'max_depth': -1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            }
            
            print("LightGBM parameters:")
            print(f"  - Objective: Binary classification")
            print(f"  - Num leaves: 50")
            print(f"  - Learning rate: 0.1")
            print(f"  - N estimators: 200")
            print(f"  - Max depth: Unlimited")
            print(f"  - Feature fraction: 0.8")
            print(f"  - Bagging fraction: 0.8")
            
            self.model = lgb.LGBMClassifier(**params)
        
        print("\nTraining model...")
        self.model.fit(
            X_train_tfidf, 
            self.y_train,
            eval_set=[(X_test_tfidf, self.y_test)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        print("✓ Training complete!")
        
        # Cross-validation
        print("\nPerforming 5-fold stratified cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, 
            X_train_tfidf, 
            self.y_train, 
            cv=skf, 
            scoring='f1',
            n_jobs=-1
        )
        print(f"CV F1 Scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_model(self, X_test_tfidf):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Predictions
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='binary'
        )
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_test': self.y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n Performance Metrics:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n Classification Report:")
        print(classification_report(
            self.y_test, 
            y_pred, 
            target_names=['Ham', 'Spam']
        ))
        
        print(f"\n Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        print(f"\n   True Negatives (Correct Ham):  {cm[0,0]}")
        print(f"   False Positives (Ham as Spam): {cm[0,1]}")
        print(f"   False Negatives (Spam as Ham): {cm[1,0]}")
        print(f"   True Positives (Correct Spam): {cm[1,1]}")
        
        # Plot confusion matrix and ROC curve
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve()
        self.plot_feature_importance()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Confusion Matrix - LightGBM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        report_path = '../reports/confusion_matrix_lgbm.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"\n Confusion matrix saved to: {report_path}")
        plt.close()
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_test, self.results['y_pred_proba'])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {self.results["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - LightGBM')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        report_path = '../reports/roc_curve_lgbm.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f" ROC curve saved to: {report_path}")
        plt.close()
    
    def plot_feature_importance(self):
        """Plot top feature importance from LightGBM"""
        print("\nGenerating feature importance plot...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = self.tfidf.get_feature_names_out()
        
        # Get top 20 features
        indices = np.argsort(importance)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Important Features - LightGBM')
        plt.tight_layout()
        
        report_path = '../reports/feature_importance_lgbm.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f" Feature importance saved to: {report_path}")
        plt.close()
        
        # Print top features
        print("\n Top 20 Most Important Features:")
        for i, idx in enumerate(indices[::-1][:20], 1):
            print(f"   {i:2d}. {feature_names[idx]:20s} - {importance[idx]:.4f}")
    
    def test_examples(self):
        """Test model on example emails"""
        print("\n" + "="*70)
        print("TESTING ON EXAMPLES")
        print("="*70)
        
        examples = [
            ("Congratulations! You have won a $1000 gift card. Click here now!", "SPAM"),
            ("Hi, can we schedule a meeting for tomorrow at 3pm?", "HAM"),
            ("URGENT: Your account will be suspended! Verify now!", "SPAM"),
            ("Thanks for the report. I'll review it and get back to you.", "HAM"),
            ("Make money fast! Work from home! No experience needed!", "SPAM"),
            ("The quarterly results are attached. Please review.", "HAM"),
        ]
        
        for text, expected in examples:
            text_tfidf = self.tfidf.transform([text])
            prediction = self.model.predict(text_tfidf)[0]
            probability = self.model.predict_proba(text_tfidf)[0]
            label = "SPAM" if prediction == 1 else "HAM"
            confidence = probability[prediction]
            
            status = "✓" if label == expected else "✗"
            print(f"\n{status} Text: {text[:60]}...")
            print(f"   Expected: {expected} | Predicted: {label} (confidence: {confidence:.4f})")
    
    def save_model(self, filename='spam_pipeline_lgbm.joblib'):
        """Save trained model and vectorizer"""
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        # Save as pipeline for easy deployment
        pipeline = {
            'tfidf': self.tfidf,
            'model': self.model
        }
        
        model_path = os.path.join('../models', filename)
        joblib.dump(pipeline, model_path)
        print(f"✓ Model pipeline saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'LightGBM',
            'vectorizer': 'TF-IDF',
            'training_date': datetime.now().isoformat(),
            'training_size': len(self.X_train),
            'test_size': len(self.X_test),
            'tfidf_features': self.tfidf.max_features,
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1': self.results['f1'],
            'roc_auc': self.results['roc_auc']
        }
        
        metadata_path = os.path.join('../models', 'model_metadata_lgbm.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("SPAM DETECTION MODEL TRAINING")
    print("LightGBM + TF-IDF")
    print("Multi-Dataset: Enron + SpamAssassin + TREC + LingSpam")
    print("="*70)
    
    # Initialize trainer
    trainer = SpamTrainerLightGBM(data_path='data\\combined_multi_spam_dataset.csv')
    
    # Load data
    if not trainer.load_data():
        return
    
    # Split data
    trainer.split_data(test_size=0.2)
    
    # Create TF-IDF features
    X_train_tfidf, X_test_tfidf = trainer.create_tfidf_features()
    
    # Train LightGBM model
    trainer.train_lightgbm(X_train_tfidf, X_test_tfidf, tune_hyperparameters=False)
    
    # Evaluate
    trainer.evaluate_model(X_test_tfidf)
    
    # Test examples
    trainer.test_examples()
    
    # Save model
    model_path = trainer.save_model()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel ready at: {model_path}")
    print("Model performance summary:")
    print(f"  • Accuracy:  {trainer.results['accuracy']*100:.2f}%")
    print(f"  • Precision: {trainer.results['precision']*100:.2f}%")
    print(f"  • Recall:    {trainer.results['recall']*100:.2f}%")
    print(f"  • F1-Score:  {trainer.results['f1']*100:.2f}%")
    print(f"  • ROC-AUC:   {trainer.results['roc_auc']:.4f}")

    print("="*70)


if __name__ == "__main__":
    main()