# ML Projects

This repository contains a small collection of machine learning projects built with Python, Jupyter notebooks, and common data science libraries. Each folder focuses on a separate prediction task or dataset.

## Projects

### 1. Ford Car Price Prediction
Path: `Ford_car_price/`

- Dataset: `ford.csv`
- Notebook: `ford_car_price.ipynb`
- Goal: predict Ford car prices from tabular vehicle data

### 2. Heart Disease Prediction
Path: `Heart_Disease/`

- Dataset: `heart.csv`
- Notebook: `Heart.ipynb`
- App: `app.py`
- Saved files: `columns.pkl`, `KNN_heart.pkl`, `scaler.pkl`
- Goal: predict the likelihood of heart disease from patient attributes

### 3. Life Insurance Prediction
Path: `Life_Insurance/`

- Dataset: `insurance.csv`
- Notebook: `insurance.ipynb`
- Goal: analyze insurance data and build a prediction workflow

### 4. Spaceship Titanic
Path: `Titanic_Spaceship/`

- Datasets: `train.csv`, `test.csv`, `Spaceship_titanic.csv`
- Notebook: `Spaceship_Titanic.ipynb`
- Goal: explore and model the Spaceship Titanic classification problem

## Tools and Libraries

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Repository Structure

```text
ML PROJECTS/
|-- Ford_car_price/
|-- Heart_Disease/
|-- Life_Insurance/
|-- Titanic_Spaceship/
`-- Readme.md
```

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/Devansh1-tech/ML-Projects.git
cd ML-Projects
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter streamlit
```

3. Open any notebook in Jupyter:

```bash
jupyter notebook
```

4. For the heart disease project, run the app if needed:

```bash
cd Heart_Disease
streamlit run app.py
```

## Purpose

This repository is intended for practice, portfolio work, and hands-on learning across multiple machine learning problems.
