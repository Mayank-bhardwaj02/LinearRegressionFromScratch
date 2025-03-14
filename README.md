# 📈 Linear Regression from Scratch

## 📌 What is Linear Regression?
Linear Regression is a **supervised learning algorithm** used for predicting a continuous value based on input data. It assumes a **linear relationship** between an independent variable (**X**) and a dependent variable (**Y**). The equation of a **simple linear regression model** is:

\[
y = mx + c
\]

where:  
- **y** = Predicted value (e.g., Salary)  
- **x** = Input variable (e.g., Years of Experience)  
- **m** = Slope (how much y changes for every unit increase in x)  
- **c** = Intercept (the y-value when x = 0)  

## 🔥 Why is Linear Regression Underrated?
Despite being one of the **simplest** machine learning models, Linear Regression is **powerful** and widely used in industries like:  
✅ **Finance** – Predicting stock prices and risk analysis  
✅ **Marketing** – Understanding ad spend vs. revenue growth  
✅ **Healthcare** – Predicting patient recovery times  
✅ **Real Estate** – Estimating house prices based on features  
✅ **Sports Analytics** – Forecasting player performance  

Linear Regression is often overlooked in favor of **complex deep learning models**, but in many real-world cases, a simple regression model can **outperform deep learning** when:
- Data is **small** and structured.
- Interpretability is **critical**.
- Computational resources are **limited**.

## 📊 The Mathematics Behind Linear Regression
The goal is to find the best-fit line that minimizes the **error** between predicted and actual values.  
To achieve this, we use **Gradient Descent** to optimize the slope (**m**) and intercept (**c**) by minimizing the **Mean Squared Error (MSE)**:

\[
MSE = \frac{1}{N} \sum (y_i - \hat{y}_i)^2
\]

where:
- \( y_i \) = Actual value  
- \( \hat{y}_i \) = Predicted value using \( y = mx + c \)  
- \( N \) = Total number of data points  

### 🏃‍♂️ **Gradient Descent Algorithm**
Gradient Descent iteratively updates **m** and **c** by computing the partial derivatives:

\[
\frac{\partial MSE}{\partial m} = -\frac{2}{N} \sum x_i (y_i - (m x_i + c))
\]

\[
\frac{\partial MSE}{\partial c} = -\frac{2}{N} \sum (y_i - (m x_i + c))
\]

The update rule:
```python
m = m - L * gradient_m
c = c - L * gradient_c
```

## 📌 Project Overview
This project implements **Linear Regression** from scratch without using libraries like Scikit-Learn, TensorFlow, or Keras. It manually computes **gradients**, applies **Gradient Descent**, and plots the **best-fit regression line** using Matplotlib. The project is designed to demonstrate **conceptual clarity** and fundamental understanding of regression.

## 📊 Features
✅ Reads dataset using Pandas  
✅ Implements **Gradient Descent** to optimize model parameters  
✅ Predicts salaries based on years of experience  
✅ Visualizes results with **Matplotlib** (Scatter plot + Regression Line)  
✅ Evaluates performance using **Mean Squared Error (MSE)**  

## 🛠️ How It Works
1. Loads the dataset (`Salary Data.csv`) containing Years of Experience and Salary.  
2. Applies **Gradient Descent** to estimate the best-fit line `y = mx + c`.  
3. Plots the **scatter plot** and overlays the **regression line**.  
4. Allows salary prediction for a given number of years of experience.  
