# Early Risk Alert AI
Author: Milton Munroe  
Course: Elements of AI – University of Helsinki  
Project Type: Probabilistic Classification (Naive Bayes)

1. Your idea in a nutshell
Project Name: Early Risk Alert AI
Early Risk Alert AI is a web-based machine learning application that predicts whether a person may be at low or high cardiovascular risk based on health indicators such as age, BMI, exercise level, blood pressure, and heart rate. The system provides a risk classification and probability score and stores prediction history in a database.

2. Background
Cardiovascular disease is one of the leading causes of death worldwide. Many individuals are unaware of their potential risk level until symptoms become serious.
This project addresses the problem of early awareness. By using health indicators that people commonly know (age, blood pressure, heart rate, BMI), the system provides an immediate estimate of risk.
The motivation behind this project is to demonstrate how AI can assist in preventative health awareness. Even simple models can help individuals better understand how lifestyle and measurable health metrics influence risk.
This topic is important because early detection and awareness can lead to better health decisions and potentially prevent serious outcomes.

3. Data and AI Techniques
This project uses a supervised machine learning classification model trained on structured health-related data.
AI techniques used:
Supervised learning
Binary classification
Probability prediction
Feature-based modeling
NumPy for feature arrays
Joblib for model loading
SQLAlchemy + PostgreSQL for storing prediction logs
The model takes numerical inputs:
Age
BMI
Exercise level
Systolic blood pressure
Diastolic blood pressure
Heart rate
The system outputs:
Risk classification (Low / High)
Probability score
The application is implemented using:
Python
Flask (web framework)
PostgreSQL (cloud database)
Deployed on Render

4. How it is used
The application is used through a web interface.
A user:
Enters their health metrics
Submits the form
Receives a risk classification and probability
The prediction is stored in a database
The user can view previous predictions in a history page
The affected users are:
Individuals interested in understanding cardiovascular risk
Students learning about AI in healthcare
Developers studying ML deployment

5. Challenges
This project does not replace medical professionals. It is not a diagnostic tool.
Limitations include:
Model accuracy depends on training data quality
Limited feature set
No integration with real medical records
No personalization beyond input variables
Risk of over-reliance on simplified predictions
AI models must be interpreted carefully and responsibly.

6. Whats next?
Future improvements could include:
Larger and more diverse datasets
Integration with wearable devices
Personalized health tracking
Mobile app version
Improved UI/UX design
Model retraining pipeline
Security and authentication
HIPAA-compliant architecture
API for mobile apps
Cloud scaling and monitoring
The project could evolve into a full preventive health monitoring platform.

7. Acknowledgments
This project was developed using:
Python
Flask
SQLAlchemy
NumPy
Joblib
PostgreSQL
Render for deployment
Open-source machine learning tools
Inspiration from introductory AI coursework and machine learning education materials.
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
