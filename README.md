# Early Risk Alert AI
Author: Milton Munroe  
Course: Elements of AI – University of Helsinki  
Project Type: Probabilistic Classification (Naive Bayes)

## 1. Project Idea in a Nutshell

Early Risk Alert is a probabilistic AI system designed to predict potential high blood pressure risk based on simple health indicators such as age, BMI, activity level, and diet patterns. The model uses Naive Bayes classification to estimate the probability of health risk.

## 2. Background

High blood pressure is a widespread global health issue that often goes undetected until complications arise. Early risk detection can significantly reduce long-term health consequences. This project explores how basic probabilistic models can assist in early health screening using structured data.

## 3. Data and AI Techniques

### Data Sources:
- Public health datasets (e.g., CDC)
- Age
- BMI
- Exercise frequency
- Diet patterns
- Family medical history

### AI Techniques Used:
- Naive Bayes classification
- Probability estimation
- Feature independence assumption
- Basic threshold optimization

Naive Bayes was chosen due to its efficiency and suitability for classification problems involving structured features.

## 4. How It Is Used

This system could be integrated into health apps, clinics, and preventative care platforms. It is intended as a decision-support tool, not a medical diagnosis system.

Stakeholders include patients, healthcare providers, and public health organizations.

## 5. Challenges

- Data bias risk  
- Assumption of feature independence  
- Limited accuracy without large datasets  
- Ethical considerations in health prediction  

## 6. Future Improvements

- Expand dataset size  
- Integrate wearable device data  
- Test logistic regression and ensemble models  
- Develop a web-based interface  

## 7. Acknowledgements

This project was inspired by:
- University of Helsinki – Elements of AI
- Public health data research

## 8. Demo Code

A simple Python implementation using Gaussian Naive Bayes is included in `demo_model.py`.

This demonstrates:
- Feature structuring
- Model training
- Risk classification
- Probability estimation

## 9. How to Run This Project

1. Clone the repository:
   git clone https://github.com/ayden611/early-risk-alert-ai.git

2. Navigate into the folder:
   cd early-risk-alert-ai

3. Install dependencies:
   pip install -r requirements.txt

4. Run the demo model:
   python demo_model.py

## 10. Model Evaluation

The model can be evaluated using:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

Future versions will include train/test split validation
and performance metrics visualization.


- Open-source Python libraries such as NumPy and scikit-learn
