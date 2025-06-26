"""
HealthTech Risk Prediction System
Main analysis script for cardiovascular disease risk prediction
Author: Manvi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class HealthTechAnalyzer:
    """
    Main class for cardiovascular disease risk prediction analysis
    """
    
    def __init__(self, data_path=None):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self, data_path=None):
        """Load and prepare the heart disease dataset"""
        if data_path:
            self.data = pd.read_csv(data_path)
        else:
            # Create synthetic heart disease dataset for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Generate synthetic healthcare data
            age = np.random.normal(54, 12, n_samples).astype(int)
            age = np.clip(age, 20, 80)
            
            sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])  # 0=female, 1=male
            
            cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.17, 0.20])  # chest pain type
            
            trestbps = np.random.normal(131, 17, n_samples).astype(int)  # resting blood pressure
            trestbps = np.clip(trestbps, 80, 200)
            
            chol = np.random.normal(246, 51, n_samples).astype(int)  # cholesterol
            chol = np.clip(chol, 120, 400)
            
            fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # fasting blood sugar
            
            restecg = np.random.choice([0, 1, 2], n_samples, p=[0.52, 0.47, 0.01])  # resting ECG
            
            thalach = np.random.normal(149, 22, n_samples).astype(int)  # max heart rate
            thalach = np.clip(thalach, 70, 220)
            
            exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])  # exercise induced angina
            
            oldpeak = np.random.exponential(1.0, n_samples)  # ST depression
            oldpeak = np.clip(oldpeak, 0, 6)
            
            slope = np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.56, 0.23])  # slope of peak exercise ST
            
            ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.59, 0.21, 0.13, 0.07])  # vessels colored by fluoroscopy
            
            thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.18, 0.36, 0.44])  # thalassemia
            
            # Create target variable with realistic correlations
            risk_score = (
                (age > 55) * 0.3 +
                sex * 0.2 +
                (cp == 0) * 0.4 +  # asymptomatic chest pain = higher risk
                (trestbps > 140) * 0.2 +
                (chol > 240) * 0.15 +
                fbs * 0.1 +
                (restecg == 1) * 0.15 +
                (thalach < 150) * 0.2 +
                exang * 0.3 +
                (oldpeak > 1) * 0.25 +
                (ca > 0) * 0.35 +
                (thal == 2) * 0.3
            )
            
            # Add some randomness
            risk_score += np.random.normal(0, 0.2, n_samples)
            target = (risk_score > 1.2).astype(int)
            
            self.data = pd.DataFrame({
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal,
                'target': target
            })
        
        print(f"Dataset loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def create_database(self, db_name='healthtech.db'):
        """Create SQLite database and store data"""
        conn = sqlite3.connect(db_name)
        
        # Create patients table
        self.data.to_sql('patients', conn, if_exists='replace', index=False)
        
        # Create some useful views
        conn.execute("""
        CREATE VIEW high_risk_patients AS
        SELECT * FROM patients 
        WHERE target = 1
        """)
        
        conn.execute("""
        CREATE VIEW age_risk_summary AS
        SELECT 
            CASE 
                WHEN age < 40 THEN 'Under 40'
                WHEN age < 50 THEN '40-49'
                WHEN age < 60 THEN '50-59'
                ELSE '60+'
            END as age_group,
            COUNT(*) as total_patients,
            SUM(target) as high_risk_count,
            ROUND(AVG(target) * 100, 2) as risk_percentage
        FROM patients
        GROUP BY age_group
        """)
        
        conn.close()
        print(f"Database '{db_name}' created successfully!")
    
    def sql_queries_demo(self, db_name='healthtech.db'):
        """Demonstrate SQL queries for data analysis"""
        conn = sqlite3.connect(db_name)
        
        print("=== SQL ANALYSIS RESULTS ===\n")
        
        # Query 1: Risk distribution by gender
        query1 = """
        SELECT 
            CASE WHEN sex = 1 THEN 'Male' ELSE 'Female' END as gender,
            COUNT(*) as total,
            SUM(target) as high_risk,
            ROUND(AVG(target) * 100, 2) as risk_percentage
        FROM patients
        GROUP BY sex
        """
        
        print("1. Risk Distribution by Gender:")
        print(pd.read_sql_query(query1, conn))
        print()
        
        # Query 2: High cholesterol patients
        query2 = """
        SELECT 
            COUNT(*) as high_chol_patients,
            SUM(target) as high_risk_count,
            ROUND(AVG(target) * 100, 2) as risk_percentage
        FROM patients
        WHERE chol > 240
        """
        
        print("2. High Cholesterol Patients (>240):")
        print(pd.read_sql_query(query2, conn))
        print()
        
        # Query 3: Age group analysis
        query3 = "SELECT * FROM age_risk_summary ORDER BY age_group"
        print("3. Risk Analysis by Age Group:")
        print(pd.read_sql_query(query3, conn))
        print()
        
        conn.close()
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("=== EXPLORATORY DATA ANALYSIS ===\n")
        
        # Basic statistics
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Heart disease cases: {self.data['target'].sum()} ({self.data['target'].mean()*100:.1f}%)")
        print()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Heart Disease Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Age distribution by target
        sns.histplot(data=self.data, x='age', hue='target', kde=True, ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Heart Disease Risk')
        axes[0,0].set_xlabel('Age')
        
        # 2. Chest pain type analysis
        chest_pain_counts = self.data.groupby(['cp', 'target']).size().unstack()
        chest_pain_counts.plot(kind='bar', ax=axes[0,1], color=['lightblue', 'salmon'])
        axes[0,1].set_title('Chest Pain Type vs Heart Disease')
        axes[0,1].set_xlabel('Chest Pain Type')
        axes[0,1].legend(['No Disease', 'Disease'])
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # 3. Cholesterol vs Max Heart Rate
        scatter = axes[0,2].scatter(self.data['chol'], self.data['thalach'], 
                                  c=self.data['target'], cmap='RdYlBu', alpha=0.6)
        axes[0,2].set_title('Cholesterol vs Max Heart Rate')
        axes[0,2].set_xlabel('Cholesterol')
        axes[0,2].set_ylabel('Max Heart Rate')
        plt.colorbar(scatter, ax=axes[0,2])
        
        # 4. Gender risk distribution
        gender_risk = self.data.groupby('sex')['target'].mean()
        gender_labels = ['Female', 'Male']
        axes[1,0].bar(gender_labels, gender_risk, color=['pink', 'lightblue'])
        axes[1,0].set_title('Heart Disease Risk by Gender')
        axes[1,0].set_ylabel('Risk Probability')
        
        # 5. Exercise induced angina
        exang_data = self.data.groupby(['exang', 'target']).size().unstack()
        exang_data.plot(kind='bar', ax=axes[1,1], color=['lightgreen', 'coral'])
        axes[1,1].set_title('Exercise Induced Angina vs Heart Disease')
        axes[1,1].set_xlabel('Exercise Induced Angina')
        axes[1,1].legend(['No Disease', 'Disease'])
        axes[1,1].tick_params(axis='x', rotation=0)
        
        # 6. Correlation heatmap
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', ax=axes[1,2])
        axes[1,2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('heart_disease_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical insights
        print("=== KEY INSIGHTS ===")
        print(f"• Average age of patients: {self.data['age'].mean():.1f} years")
        print(f"• Male patients: {(self.data['sex'].sum()/len(self.data)*100):.1f}%")
        print(f"• High cholesterol (>240): {(self.data['chol'] > 240).sum()} patients")
        print(f"• Exercise induced angina: {(self.data['exang'].sum()/len(self.data)*100):.1f}%")
        print()
    
    def prepare_data(self):
        """Prepare data for machine learning"""
        # Features and target
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data prepared! Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("=== TRAINING MACHINE LEARNING MODELS ===\n")
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for Logistic Regression and SVM
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(self.X_test_scaled if name in ['Logistic Regression', 'SVM'] else self.X_test, self.y_test)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✓ {name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        print("\nModel training completed!")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION RESULTS ===\n")
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16, fontweight='bold')
        
        # Performance comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        axes[0,0].bar(x + width/2, auc_scores, width, label='AUC Score', color='lightcoral')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(model_names, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # ROC Curves
        for name in model_names:
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['probabilities'])
            axes[0,1].plot(fpr, tpr, label=f"{name} (AUC = {self.results[name]['auc_score']:.3f})")
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Feature importance (Random Forest)
        if 'Random Forest' in self.models:
            feature_importance = self.models['Random Forest'].feature_importances_
            feature_names = self.X_train.columns
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1][:10]
            
            axes[1,0].bar(range(len(indices)), feature_importance[indices], color='lightgreen')
            axes[1,0].set_xlabel('Features')
            axes[1,0].set_ylabel('Importance')
            axes[1,0].set_title('Top 10 Feature Importance (Random Forest)')
            axes[1,0].set_xticks(range(len(indices)))
            axes[1,0].set_xticklabels([feature_names[i] for i in indices], rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # Confusion Matrix for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        axes[1,1].set_title(f'Confusion Matrix - {best_model_name}')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print("Detailed Model Performance:")
        print("-" * 50)
        for name in model_names:
            print(f"\n{name}:")
            print(f"  Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"  AUC Score: {self.results[name]['auc_score']:.4f}")
            print(f"  Classification Report:")
            print(classification_report(self.y_test, self.results[name]['predictions'], 
                                      target_names=['No Disease', 'Disease'], digits=3))
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Models and scaler saved successfully!")
    
    def predict_risk(self, patient_data):
        """Predict risk for new patient data"""
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        # Prepare data
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Scale if necessary
        if best_model_name in ['Logistic Regression', 'SVM']:
            patient_scaled = self.scaler.transform(patient_df)
            risk_probability = best_model.predict_proba(patient_scaled)[0][1]
        else:
            risk_probability = best_model.predict_proba(patient_df)[0][1]
        
        risk_level = "HIGH" if risk_probability > 0.5 else "LOW"
        
        return {
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'model_used': best_model_name
        }

def main():
    """Main execution function"""
    print("🏥 HealthTech Risk Prediction System")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = HealthTechAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    
    # Create database
    analyzer.create_database()
    
    # SQL analysis
    analyzer.sql_queries_demo()
    
    # EDA
    analyzer.exploratory_data_analysis()
    
    # Prepare data
    analyzer.prepare_data()
    
    # Train models
    analyzer.train_models()
    
    # Evaluate models
    analyzer.evaluate_models()
    
    # Save models
    analyzer.save_models()
    
    # Example prediction
    print("\n=== EXAMPLE RISK PREDICTION ===")
    sample_patient = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    
    prediction = analyzer.predict_risk(sample_patient)
    print(f"Patient Risk Assessment:")
    print(f"  • Risk Probability: {prediction['risk_probability']:.2%}")
    print(f"  • Risk Level: {prediction['risk_level']}")
    print(f"  • Model Used: {prediction['model_used']}")
    
    print("\n✅ Analysis completed! Check the generated visualizations and saved models.")
    print("🚀 Ready for GitHub deployment!")

if __name__ == "__main__":
    main()
