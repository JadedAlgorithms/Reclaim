# Reclaim: Predictive Recovery for Failed Digital Transactions

## Abstract
Failed digital transactions represent a significant source of revenue leakage in modern payment processing systems. The Reclaim project presents a machine learning framework designed to predict the optimal recovery intervention for failed transactions based on contextual features, including the underlying failure reason, customer risk tier, and historical recovery success rates. By formulating transaction recovery as a predictive ranking problem, Reclaim aims to maximize revenue recovery while minimizing suboptimal retry attempts.

## Methodology
The system architecture relies on a synthetic data pipeline that simulates transaction outcomes using probabilistic rules. The framework comprises the following core components:

### 1. Data Generation and Simulation
The simulation engine generates synthetic customer profiles and transaction records. Recovery attempts are simulated using a context aware probabilistic model. The base probability of success is modulated by the selected intervention, the underlying failure reason, the customer risk tier, and the temporal decay associated with sequential attempts.

### 2. Feature Engineering
The feature extraction pipeline constructs a dataset optimized for predictive modeling. Key predictive features include the transaction amount, payment method, failure reason, risk tier, attempt number, elapsed time since failure, and the customer historical success rate. To prevent data leakage, historical success rates are computed using a shifted expanding window methodology.

### 3. Predictive Modeling
The framework evaluates classification algorithms, primarily Logistic Regression and Random Forest models. These models are trained to predict the binary outcome of a recovery attempt. The Logistic Regression model, utilizing balanced class weights and standardized features, demonstrated superior performance in maximizing total recovered revenue by prioritizing recall.

### 4. Recommendation Agent
The Reclaim Agent serves as the inference engine. Given a failed transaction, the agent generates feature vectors for all candidate recovery actions and evaluates them using the trained model. The candidate actions are ranked by their predicted probability of success, providing a quantitative recommendation matrix to the merchant.

## Evaluation Metrics
The primary evaluation metric is the Revenue Recovery Rate, defined as the ratio of recovered monetary value to the total monetary value at risk. In empirical evaluations, the Reclaim predictive agent significantly outperformed static baseline strategies, demonstrating a substantial uplift in aggregate recovered revenue.

## Repository Structure
* src/entities.py: Data structures representing customers and transactions.
* src/generate_data.py: Pipeline for synthetic data generation.
* simulator/recovery_simulator.py: Probabilistic outcome simulation logic.
* src/features.py: Feature engineering and dataset construction.
* src/train.py: Model training, evaluation, and persistence.
* src/agent.py: Inference agent for generating ranked recommendations.
* src/eda.py: Exploratory data analysis and visualization scripts.
* notebooks/demo.ipynb: Interactive demonstration of the analytical pipeline.

## Conclusion
The Reclaim framework provides a quantitative, data driven approach to transaction recovery. By leveraging contextual signals, the system transitions recovery strategies from static heuristics to dynamic, predictive interventions, offering a robust solution for revenue optimization in payment gateways.
