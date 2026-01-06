# 📊 S&P 500 Stock Market Analysis

This project provides an **end-to-end analysis of S&P 500 stock market data**, combining **data collection**, **processing**, **exploratory analysis**, and **interactive visualization**.  
Historical stock prices are collected using **Python** and **Yahoo Finance**, structured into **CSV datasets**, and explored through both a **Python interactive dashboard** and a **Power BI dashboard**.

The project is designed as a solid foundation for **future Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) applications** in financial market prediction.

---

## 🚀 Project Overview

The main objectives of this project are to:

- Scrape the list of **S&P 500 companies** from Wikipedia.
- Download **historical stock price data** from Yahoo Finance.
- Clean, normalize, and structure the data using Python.
- Compute **technical indicators** such as rolling moving averages.
- Export structured datasets to **CSV format**.
- Build **interactive dashboards** using:
  - Python (Matplotlib + ipywidgets)
  - Power BI (advanced analytics and reporting)

---

## 🛠️ Technologies Used

### Programming & Data Analysis
- **Python 3.11.5**
- Libraries:
  - `requests`
  - `BeautifulSoup`
  - `pandas`
  - `numpy`
  - `yfinance`
  - `matplotlib`
  - `seaborn`
  - `ipywidgets`

### Visualization
- **Python Interactive Dashboard**
- **Power BI**

### Tools
- **Git & GitHub** – version control
- **Yahoo Finance** – historical stock data source

---

## 📂 Folder Structure


S-P-500-Stock-Market-Analysis/
- ├── Dashboard-Power-BI-Cap/ # Power BI dashboard screenshots
- ├── Dashboard-Python-Cap/ # Python interactive dashboard screenshots
- ├── Pictures/ # Images used in Python dashboards and documentation
- ├── S&P-500-Stock-Market-Analysis-Project.ipynb  │  # Jupyter Notebook (interactive analysis)
- ├── S&P-500-Stock-Market-Analysis-Project.py  │  # Python script (scraping, preprocessing, dashboards)
- ├── dark_blue.json  │  # Custom Power BI theme file
- ├── README.md  │  # Project documentation



---

## 📊 Data Description

The project generates **three main datasets**, all exported to CSV:

- **symbols_df.csv**  
  Contains the list of S&P 500 companies (ID, symbol, company name).

- **historical_prices_df.csv**  
  Stores historical OHLCV data (Open, High, Low, Close, Volume).

- **calculated_metrics_df.csv**  
  Contains derived indicators such as rolling moving averages (MA-1 to MA-50).


---



## 🏆 Results & Dashboards

### 🖥️ Python Interactive Dashboard
The Python dashboard provides:
- Dynamic stock symbol selection
- Yearly / monthly trading volume visualization
- Stock price comparison
- Moving average analysis

**Example (Python Dashboard):**

### 📈 Key Insights
- **Market Activity Trends:**
  Yearly and monthly trading volume analysis highlights periods of increased market activity.

- **Stock Performance Comparison:**
  Interactive comparison of price evolution across different companies.

- **Trend Smoothing:**  
  Moving averages illustrate long-term trends and reduce short-term market noise.

![image][Python Dashboard]([https://github.com/YoussefKamm/Car-Price-Predictor/blob/main/Screenshot-Web.png](https://github.com/YoussefKamm/S-P-500-Stock-Market-Analysis/tree/main/Dashboard-Python-Cap))


---

## 📊 Power BI Dashboard

The Power BI dashboard offers a **professional analytics view** with advanced filtering and reporting.

### 📈 Key Insights
- **Market Trends:**
  Identify sectors driving market growth.
- **Stock Volatility:**
  Measure historical volatility across different time periods.
- **Performance Comparison:**
  Compare individual stocks with the overall S&P 500 index.

#### Home Dashboard
![Power BI Home]([https://github.com/YoussefKamm/S-P-500-Stock-Market-Analysis/blob/main/Dashboard-Power BI-Cap/Home.jpg](https://github.com/YoussefKamm/S-P-500-Stock-Market-Analysis/blob/main/Dashboard-Power%20BI-Cap/Analysis.jpg))

#### Analysis Dashboard
![Power BI Analysis](https://github.com/YoussefKamm/S-P-500-Stock-Market-Analysis/blob/main/Dashboard-Power BI-Cap/Analysis.jpg)

---

## 🤖 Future Improvements (AI & Machine Learning Focus)

Future development of this project will focus on **Artificial Intelligence and predictive modeling**, including:

- Building **machine learning models** for stock price prediction and trend classification.
- Applying **deep learning techniques** (LSTM, GRU, Transformer models) for time-series forecasting.
- Feature engineering using technical indicators and historical patterns.
- Model evaluation and backtesting on historical market data.
- Extending the pipeline toward **automated decision-support systems** for financial analysis.

---

## 🤝 Contributing

Contributions are welcome!

- Fork the repository
- Create a new branch
- Submit a pull request
- Open issues for suggestions or improvements

---

## 👤 Author

**Youssef Kammoun**  
- [LinkedIn : Youssef Kammoun](https://www.linkedin.com/in/kammounyoussef)
- [Email : kammounyoussef54@gmail.com](mailto:kammounyoussef54@gmail.com) 
