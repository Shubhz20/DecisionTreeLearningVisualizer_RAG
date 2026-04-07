import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Decision Tree Explorer", page_icon="🌲", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e5d6c;
    }
    div[data-testid="stExpander"] {
        background-color: #1e2130;
        border: 1px solid #4e5d6c;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def generate_data(dataset_type, noise=0.1, n_samples=300):
    if dataset_type == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=noise*5, random_state=42)
    else:
        # Simple Linear-ish split
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise*3, random_state=42)
    return pd.DataFrame(X, columns=["X1", "X2"]), y

def calculate_gini(y):
    if len(y) == 0: return 0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p**2)

def calculate_entropy(y):
    if len(y) == 0: return 0
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def plot_decision_boundary(model, X, y, resolution=0.1):
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    fig.add_trace(go.Contour(x=np.arange(x_min, x_max, resolution),
                             y=np.arange(y_min, y_max, resolution),
                             z=Z, colorscale='Viridis', opacity=0.3, showscale=False))
    
    fig.add_trace(go.Scatter(x=X.iloc[:,0], y=X.iloc[:,1], mode='markers',
                             marker=dict(color=y, colorscale='Viridis', line_width=1),
                             text=[f"Class: {i}" for i in y]))
    
    fig.update_layout(title="Decision Regions", xaxis_title="X1", yaxis_title="X2", 
                      height=500, template="plotly_dark")
    return fig

# Sidebar
st.sidebar.title("🌲 Decision Tree Explorer")
st.sidebar.markdown("---")
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Moons", "Circles", "Blobs", "Linear"])
noise_val = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
n_samples = st.sidebar.slider("Sample Size", 50, 1000, 300)

st.sidebar.markdown("### Hyperparameters")
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 50, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 50, 1)

# Generate Data
X, y = generate_data(dataset_choice, noise_val, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Main Content
st.title("Interactive Decision Tree Learning Visualizer")
st.markdown("Explore how Decision Trees grow, split data, and make decisions through interactive visualizations.")

tabs = st.tabs([
    "1. Data & Splits", 
    "2. Impurity Measures", 
    "3. Split Selection", 
    "4. Tree Structure", 
    "5. Overfitting & Depth", 
    "6. Prediction Path",
    "7. Noise & Pruning"
])

# Tab 1: Data & Splits
with tabs[0]:
    st.header("Data Partitioning & Feature Splits")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Decision Trees split data using horizontal and vertical lines (axis-aligned).")
        feat = st.selectbox("Split Feature", ["X1", "X2"])
        threshold = st.slider(f"Threshold for {feat}", float(X[feat].min()), float(X[feat].max()), float(X[feat].mean()))
        
        # Calculate split stats
        mask = X[feat] <= threshold
        y_left = y[mask]
        y_right = y[~mask]
        
        st.info(f"Left Subset: {len(y_left)} samples")
        st.info(f"Right Subset: {len(y_right)} samples")
        
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X["X1"], y=X["X2"], mode='markers',
                                 marker=dict(color=y, colorscale='Viridis', showscale=False)))
        
        if feat == "X1":
            fig.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="orange")
        else:
            fig.add_hline(y=threshold, line_width=3, line_dash="dash", line_color="orange")
            
        fig.update_layout(title=f"Manual Split Visualization ({feat} = {threshold:.2f})", 
                          xaxis_title="X1", yaxis_title="X2", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Impurity Measures
with tabs[1]:
    st.header("Gini Impurity vs Entropy")
    p = np.linspace(0.001, 0.999, 100)
    gini = 1 - (p**2 + (1-p)**2)
    entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p)) / 2 # Normalized entropy for visual comparison
    
    st.markdown("""
    Impurity measures how 'mixed' the classes are in a node. 
    - **Gini**: $1 - \sum p_i^2$
    - **Entropy**: $-\sum p_i \log_2(p_i)$
    """)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=gini, name="Gini Impurity", line=dict(color='cyan', width=3)))
    fig.add_trace(go.Scatter(x=p, y=entropy, name="Normalized Entropy", line=dict(color='magenta', width=3)))
    fig.update_layout(title="Impurity Measures Comparison", xaxis_title="Probability of Class 1", 
                      yaxis_title="Impurity Value", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Split Selection
with tabs[2]:
    st.header("Finding the Best Split")
    st.write("The algorithm iterates through all features and all possible thresholds to find the one that maximizes Information Gain (reduction in impurity).")
    
    selected_feat = st.radio("Choose Feature to analyze", ["X1", "X2"], horizontal=True)
    
    thresholds = np.linspace(X[selected_feat].min(), X[selected_feat].max(), 50)
    gains = []
    
    parent_impurity = calculate_gini(y) if criterion == "gini" else calculate_entropy(y)
    
    for t in thresholds:
        left_mask = X[selected_feat] <= t
        y_l, y_r = y[left_mask], y[~left_mask]
        
        if len(y_l) == 0 or len(y_r) == 0:
            gains.append(0)
            continue
            
        w_l = len(y_l) / len(y)
        w_r = len(y_r) / len(y)
        
        child_impurity = w_l * (calculate_gini(y_l) if criterion == "gini" else calculate_entropy(y_l)) + \
                         w_r * (calculate_gini(y_r) if criterion == "gini" else calculate_entropy(y_r))
        
        gains.append(parent_impurity - child_impurity)
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=gains, mode='lines+markers', name="Impurity Reduction",
                             line=dict(color='lime')))
    best_t = thresholds[np.argmax(gains)]
    fig.add_vline(x=best_t, line_color="yellow", line_dash="dash", annotation_text=f"Best Split: {best_t:.2f}")
    
    fig.update_layout(title=f"Information Gain vs Threshold for {selected_feat}", 
                      xaxis_title="Threshold", yaxis_title="Gain", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Tree Structure
with tabs[3]:
    st.header("Hierarchical Tree Growth")
    
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)
    
    try:
        dot_data = export_graphviz(clf, out_file=None, 
                                   feature_names=["X1", "X2"],  
                                   class_names=[str(i) for i in np.unique(y)],  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        st.graphviz_chart(dot_data)
    except Exception as e:
        st.warning("Graphviz 'dot' binary not found. Falling back to Matplotlib visualization.")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(clf, feature_names=["X1", "X2"], class_names=[str(i) for i in np.unique(y)], 
                  filled=True, rounded=True, ax=ax)
        st.pyplot(fig)
    
    st.write(f"Training Accuracy: {accuracy_score(y_train, clf.predict(X_train)):.2f}")
    st.write(f"Testing Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2f}")

# Tab 5: Overfitting & Depth
with tabs[4]:
    st.header("The Effect of Depth on Overfitting")
    
    depths = list(range(1, 15))
    train_acc, test_acc = [], []
    
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42)
        m.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, m.predict(X_train)))
        test_acc.append(accuracy_score(y_test, m.predict(X_test)))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depths, y=train_acc, name="Train Accuracy", line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=depths, y=test_acc, name="Test Accuracy", line=dict(color='orange')))
    fig.update_layout(title="Accuracy vs Tree Depth", xaxis_title="Max Depth", 
                      yaxis_title="Accuracy", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Decision Boundaries")
    d_select = st.slider("Select Depth to visualize boundaries", 1, 15, max_depth)
    m_viz = DecisionTreeClassifier(max_depth=d_select, random_state=42)
    m_viz.fit(X, y)
    st.plotly_chart(plot_decision_boundary(m_viz, X, y), use_container_width=True)

# Tab 6: Prediction Path
with tabs[5]:
    st.header("Tracing a Prediction")
    st.write("Pick a point from the data to see the decisions made at each node.")
    
    sample_idx = st.number_input("Sample Index", 0, len(X)-1, 0)
    sample = X.iloc[sample_idx:sample_idx+1]
    
    # Train model (using current sidebar params)
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, 
                                 min_samples_leaf=min_samples_leaf)
    clf.fit(X, y)
    
    pred = clf.predict(sample)[0]
    path = clf.decision_path(sample)
    
    st.write(f"**Instance Features:** X1={sample.iloc[0,0]:.2f}, X2={sample.iloc[0,1]:.2f}")
    st.success(f"**Predicted Class:** {pred}")
    
    # Extract rules
    node_indicator = clf.decision_path(sample)
    leaf_id = clf.apply(sample)
    node_index = node_indicator.indices[node_indicator.indptr[0]:
                                        node_indicator.indptr[1]]
                                        
    st.markdown("### Decision Rules Path:")
    for node_id in node_index:
        if leaf_id[0] == node_id:
            st.markdown(f"🚩 **Leaf Node {node_id} reached**")
            continue
            
        feature = clf.tree_.feature[node_id]
        threshold = clf.tree_.threshold[node_id]
        val = sample.values[0, feature]
        
        if val <= threshold:
            symbol = "≤"
        else:
            symbol = ">"
            
        st.markdown(f"- Node {node_id}: Feature `X{feature+1}` ({val:.2f}) {symbol} {threshold:.2f}")

# Tab 7: Noise & Pruning
with tabs[6]:
    st.header("Noise, Depth & Pruning")
    
    noise_lvl = st.slider("Add random noise to labels (%)", 0, 50, 10, key="noise_prune")
    y_noisy = y.copy()
    n_noisy = int(len(y) * (noise_lvl / 100))
    noise_indices = np.random.choice(len(y), n_noisy, replace=False)
    # Flip or change labels
    unique_labels = np.unique(y)
    for idx in noise_indices:
        y_noisy[idx] = np.random.choice([l for l in unique_labels if l != y[idx]])
        
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Deep Tree (Overfitted)")
        m_deep = DecisionTreeClassifier(max_depth=15)
        m_deep.fit(X, y_noisy)
        st.plotly_chart(plot_decision_boundary(m_deep, X, y_noisy), use_container_width=True)
        st.write(f"Accuracy: {accuracy_score(y_noisy, m_deep.predict(X)):.2f}")
        
    with colB:
        st.subheader("Pruned Tree (Generalizing)")
        ccp_alpha = st.slider("Pruning Alpha (CCP)", 0.0, 0.1, 0.01, step=0.005)
        m_pruned = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        m_pruned.fit(X, y_noisy)
        st.plotly_chart(plot_decision_boundary(m_pruned, X, y_noisy), use_container_width=True)
        st.write(f"Accuracy: {accuracy_score(y_noisy, m_pruned.predict(X)):.2f}")

st.markdown("---")
st.markdown("Created by Antigravity for RAG Class Activity.")
