# 📈 Linear Regression with Gradient Descent

This repository contains a Python implementation of a simple linear regression model for predicting employee salaries based on their years of experience. The model is trained using **Batch Gradient Descent**, a fundamental optimization algorithm in machine learning.

The project is designed as a clear, step-by-step example to understand how linear regression and gradient descent work from the ground up, without relying on high-level libraries like `scikit-learn` for the core algorithm.

## ✨ Features

* **Custom Implementation**: The core functions for calculating the cost function and the gradient are implemented from scratch.

* **Gradient Descent**: The program uses a batch gradient descent algorithm to find the optimal model parameters ($w$ and $b$).

* **Data Visualization**: Matplotlib is used to visualize the training data and the final linear fit, providing a clear understanding of the model's performance.

## 📁 Files

* `linear_regression.ipynb`: The Jupyter Notebook containing the code that loads the data, defines the linear regression model functions, trains the model, and visualizes the results.

* `data.csv`: The dataset used for training, containing 'exp (in months)' and 'salary (in thousands)'.

## 🚀 How to Run

### Prerequisites

You need to have Python and the following libraries installed:

* `numpy`

* `pandas`

* `matplotlib`

You can install them using `pip`:


pip install numpy pandas matplotlib


### Execution

To run the program, you can open the `linear_regression.ipynb` file in a Jupyter environment (like Jupyter Notebook, JupyterLab, or VS Code with the Python extension) and run the cells.

## 📊 Results

The program outputs the final optimized parameters ($w$ and $b$) and visualizes the fitted line. The normalization step is crucial, as it allows for a more efficient training process, leading to a converged model in fewer iterations.

The final predicted salary for a new data point is also calculated and printed, demonstrating the model's practical application.

## 📐 The Math Behind the Model

### 1. The Model Function: $$f_{w,b}(x) = wx + b$$

This is the linear equation that represents our model. It's a straight line where:

* $x$ is the input feature (Experience).

* $w$ is the **weight** or **slope** of the line. It determines how much the salary increases for each unit increase in experience.

* $b$ is the **bias** or **y-intercept**. It's the base salary when experience is zero.

* $f_{w,b}(x)$ is the predicted output (Salary).

### 2. The Cost Function: $$J(w,b) = \frac{1}{2m} \sum_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

The cost function measures how "wrong" our model is. It calculates the squared difference between the model's prediction ($f_{w,b}(x)$) and the actual salary ($y$) for each data point. The goal is to find the values of $w$ and $b$ that minimize this cost. Minimizing the cost means our model's predictions are as close as possible to the actual data points.

### 3. Gradient Descent:

**Gradient descent** is an iterative optimization algorithm used to find the minimum of the cost function. It repeatedly adjusts the parameters $w$ and $b$ in the direction that decreases the cost the most.

The algorithm uses the partial derivatives of the cost function with respect to $w$ and $b$ to determine this direction:

$$\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})$$

$$\frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) -y^{(i)})x^{(i)}$$

These derivatives are the "gradient" of the cost function. With each iteration, the parameters are updated using the following rules:

$w := w - \alpha \frac{\partial J(w,b)}{\partial w}$

$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$

Here, $\alpha$ is the **learning rate**, a small number that controls the size of each step. The process continues until the parameters converge to values that give the lowest possible cost.
