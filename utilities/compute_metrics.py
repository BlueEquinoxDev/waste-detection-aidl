from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def create_compute_metrics(label_names, logdir):
    def compute_metrics(eval_pred, logdir = logdir):
        """Compute metrics for classification task"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        #print(f"Predictions: {predictions} - Labels: {labels}")

        # Overall accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # Detailed classification report
        report = classification_report(
            labels, 
            predictions, 
            target_names=label_names,
            output_dict=True
        )

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Create confusion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            xticklabels=label_names,
            yticklabels=label_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs(f'{logdir}/plots', exist_ok=True)
        plt.savefig(f'{logdir}/plots/confusion_matrix.png')
        plt.close()
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Add per-class metrics
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                metrics[f'{class_name}_precision'] = class_metrics['precision']
                metrics[f'{class_name}_recall'] = class_metrics['recall']
                metrics[f'{class_name}_f1'] = class_metrics['f1-score']
        
        return metrics
    return compute_metrics