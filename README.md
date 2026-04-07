# Decision Tree Interactive Explorer

This Streamlit application is designed to help students understand the inner workings of Decision Trees.

## Features
1. **Data Partitioning**: Visualize manual splits on synthetic datasets (Moons, Circles, Blobs).
2. **Impurity Measures**: Compare Gini Impurity vs. Entropy.
3. **Split Selection**: See how Information Gain is calculated across different thresholds.
4. **Tree Structure**: Explore the hierarchical growth of the tree.
5. **Overfitting & Depth**: Observe how tree depth affects decision boundaries and accuracy.
6. **Prediction Path**: Trace the decision rules for a specific data point.
7. **Noise & Pruning**: See the effect of label noise and how cost-complexity pruning (CCP) helps.

## How to Run
1. Ensure you have Python 3.10+ installed.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Tech Stack
- **Dashboard**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Visualization**: Plotly, Matplotlib, Graphviz
- **Data**: NumPy, Pandas
