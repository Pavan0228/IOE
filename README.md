# Market Basket Analysis Dashboard 🛒

An interactive dashboard for analyzing store transaction data, discovering product associations, and uncovering shopping patterns using association rule mining algorithms (Apriori & FP-Growth).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

-   **📊 Overview Dashboard**: Dataset statistics, top products, and transaction samples
-   **🔍 Association Rules Mining**: Discover "if X then Y" patterns using Apriori or FP-Growth
-   **🕸️ Network Visualization**: Interactive product association networks
-   **🔥 Product Insights**: Co-occurrence heatmaps and detailed statistics
-   **⏰ Time-based Analysis**: Shopping trends by hour, day, and date

## 📁 Project Structure

```
IOE/
├── dashboard.py                    # Main UI file (Streamlit interface)
├── market_basket_analysis.py      # Business logic module
├── visualization.py                # Visualization functions
├── requirements.txt                # Python dependencies
├── thingspeak_ready.csv           # Transaction data
└── RFID/                          # RFID Smart Basket code
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Pavan0228/IOE.git
    cd IOE
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the dashboard**

    ```bash
    python -m streamlit run dashboard.py
    ```

4. **Open your browser** at `http://localhost:8501`

## 💡 How It Works

### Architecture

The project follows a **clean separation of concerns** with three main modules:

1. **Business Logic** (`market_basket_analysis.py`)

    - `MarketBasketAnalyzer`: Core analysis class
    - `DataValidator`: Parameter validation
    - `RuleInterpreter`: Results interpretation

2. **Visualization** (`visualization.py`)

    - `MarketBasketVisualizer`: All plotting functions
    - `MetricsDisplay`: Metric formatting utilities

3. **User Interface** (`dashboard.py`)
    - Streamlit web application
    - Interactive controls and tabs
    - Data flow orchestration

### Algorithms

#### Apriori

-   Classic frequent itemset mining algorithm
-   Best for smaller datasets (<10K transactions)
-   Well-tested and reliable

#### FP-Growth

-   Fast Pattern Growth algorithm
-   Optimal for large datasets (>10K transactions)
-   Uses less memory and faster execution

## 📊 Understanding the Results

### Association Rules

Rules follow the format: **IF [Antecedents] THEN [Consequents]**

**Example**: `IF {milk, bread} THEN {butter}`

**Metrics**:

-   **Support** (0.05): Appears in 5% of transactions
-   **Confidence** (0.70): 70% of customers buying milk & bread also buy butter
-   **Lift** (2.5): 2.5x more likely to buy butter with milk & bread
-   **Conviction**: Strength of implication

### Lift Interpretation

| Lift Value | Meaning                     |
| ---------- | --------------------------- |
| > 1.5      | Strong positive correlation |
| 1.0 - 1.5  | Weak positive correlation   |
| = 1.0      | Independent products        |
| < 1.0      | Negative correlation        |

## 🎯 Business Applications

1. **🏪 Product Placement**: Place associated items near each other
2. **🎁 Bundle Offers**: Create promotions for frequently co-purchased products
3. **📦 Inventory Management**: Stock complementary products together
4. **💰 Cross-selling**: Recommend products based on cart contents
5. **👥 Staffing**: Schedule more staff during peak hours
6. **📈 Marketing**: Target campaigns based on product associations

## ⚙️ Configuration

### Sidebar Controls

-   **Mining Algorithm**: Choose between Apriori or FP-Growth
-   **Minimum Support**: 0.001 - 0.1 (lower = more patterns, slower)
-   **Rule Metric**: Confidence, Lift, or Support
-   **Minimum Threshold**: Filter rules by strength

### Recommended Settings

For initial exploration:

-   Algorithm: FP-Growth
-   Minimum Support: 0.01 (1%)
-   Metric: Lift
-   Minimum Threshold: 1.2

## 📸 Screenshots

### Dashboard Overview

Interactive dashboard with 5 main tabs for comprehensive analysis

### Association Rules

Discover and explore product relationships with sortable tables

### Network Visualization

Interactive network graphs showing product associations

### Time Trends

Analyze shopping patterns across different time periods

## 🛠️ Tech Stack

-   **Python 3.8+**
-   **Streamlit**: Web application framework
-   **Pandas**: Data manipulation
-   **Plotly**: Interactive visualizations
-   **NetworkX**: Network graph analysis
-   **MLxtend**: Machine learning extensions (Apriori, FP-Growth)
-   **NumPy**: Numerical computations

## 📝 Data Format

Your CSV file should have this structure:

```csv
created_at,field1,field2,field3,field4,field5,field6,field7,field8
2025-10-16T10:00:00,product1,product2,product3,,,,,
2025-10-16T10:01:00,product4,product5,,,,,,
```

-   **created_at**: Transaction timestamp (ISO 8601 format)
-   **field1-field8**: Products purchased (empty fields allowed)

## 🔧 Advanced Usage

### Programmatic Access

```python
from market_basket_analysis import MarketBasketAnalyzer

# Initialize analyzer
analyzer = MarketBasketAnalyzer('thingspeak_ready.csv')
analyzer.load_data()
analyzer.prepare_transactions()

# Get statistics
stats = analyzer.get_dataset_stats()

# Run analysis
analyzer.create_basket_matrix()
frequent_itemsets = analyzer.get_frequent_itemsets(min_support=0.01)
rules = analyzer.generate_association_rules(metric='lift', min_threshold=1.2)

# Get insights
top_products = analyzer.get_top_products(top_n=10)
network = analyzer.build_association_network(top_n=50)
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Business logic goes in `market_basket_analysis.py`
2. Visualizations go in `visualization.py`
3. UI components go in `dashboard.py`
4. Maintain separation of concerns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Pavan Rasal**

-   GitHub: [@Pavan0228](https://github.com/Pavan0228)
-   Website: [pavanrasal.me](https://pavanrasal.me)

## 🙏 Acknowledgments

-   Built with [Streamlit](https://streamlit.io/)
-   Market basket analysis algorithms from [MLxtend](http://rasbt.github.io/mlxtend/)
-   Visualizations powered by [Plotly](https://plotly.com/)

## 📞 Support

For issues, questions, or suggestions:

-   🐛 [Open an issue](https://github.com/Pavan0228/IOE/issues)
-   📧 Contact via [GitHub](https://github.com/Pavan0228)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
