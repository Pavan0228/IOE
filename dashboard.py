"""
Market Basket Analysis Dashboard - UI Module

This is the main UI file that uses the business logic and visualization modules
to create an interactive Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
from market_basket_analysis import MarketBasketAnalyzer, RuleInterpreter
from visualization import MarketBasketVisualizer

# Page configuration
st.set_page_config(
    page_title="Market Basket Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2ca02c;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_analyzer(file_path):
    """Get or create analyzer instance"""
    return MarketBasketAnalyzer(file_path)


@st.cache_data
def load_and_prepare_data(_analyzer):
    """Load and prepare data using the analyzer"""
    _analyzer.load_data()
    _analyzer.prepare_transactions()
    return _analyzer.df, _analyzer.transactions


@st.cache_data
def run_analysis(_analyzer, min_support, algorithm, metric, min_threshold):
    """Run the complete market basket analysis"""
    _analyzer.create_basket_matrix()
    frequent_itemsets = _analyzer.get_frequent_itemsets(min_support, algorithm)
    rules = _analyzer.generate_association_rules(metric, min_threshold)
    return frequent_itemsets, rules


def main():
    st.title("🛒 Market Basket Analysis Dashboard")
    st.markdown("### Uncover Shopping Patterns and Product Associations")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Load data
    data_file = "thingspeak_ready.csv"
    
    try:
        analyzer = get_analyzer(data_file)
        df, transactions = load_and_prepare_data(analyzer)
        st.sidebar.success(f"✅ Data loaded: {len(df)} transactions")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Get dataset statistics
    stats = analyzer.get_dataset_stats()
    
    # Display basic stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.metric("Total Transactions", stats['total_transactions'])
    st.sidebar.metric("Unique Products", stats['unique_products'])
    st.sidebar.metric("Avg Items per Transaction", f"{stats['avg_items_per_transaction']:.2f}")
    
    # Algorithm selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Algorithm Settings")
    
    algorithm = st.sidebar.selectbox(
        "Mining Algorithm",
        ["apriori", "fpgrowth"],
        help="Apriori: Classic algorithm, slower but well-tested. FP-Growth: Faster for large datasets."
    )
    
    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Minimum frequency of itemset occurrence (lower = more rules, but slower)"
    )
    
    metric = st.sidebar.selectbox(
        "Rule Metric",
        ["confidence", "lift", "support"],
        index=1,
        help="Metric for filtering association rules"
    )
    
    min_threshold = st.sidebar.slider(
        "Minimum Threshold",
        min_value=0.1 if metric == "support" else 0.5 if metric == "confidence" else 1.0,
        max_value=1.0 if metric in ["support", "confidence"] else 5.0,
        value=0.5 if metric in ["support", "confidence"] else 1.2,
        step=0.1,
        help="Minimum value for the selected metric"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔍 Association Rules", 
        "🕸️ Network Analysis",
        "🔥 Product Insights",
        "⏰ Time Trends"
    ])
    
    # Initialize visualizer
    visualizer = MarketBasketVisualizer()
    
    # Tab 1: Overview
    with tab1:
        st.header("Overview & Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📦 Total Transactions", f"{stats['total_transactions']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🏷️ Unique Products", f"{stats['unique_products']:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🛍️ Avg Items/Transaction", f"{stats['avg_items_per_transaction']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📅 Date Range", f"{stats['date_range_days']} days")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top products visualization
        st.subheader("🏆 Most Popular Products")
        top_n = st.slider("Number of products to display", 10, 50, 20, key="overview_top_n")
        
        products_df = analyzer.get_top_products(top_n)
        fig_top = visualizer.plot_top_products(products_df, top_n)
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Sample transactions
        st.markdown("---")
        st.subheader("📋 Sample Transactions")
        sample_size = min(10, len(transactions))
        for i, transaction in enumerate(transactions[:sample_size], 1):
            st.write(f"**Transaction {i}:** {', '.join(transaction)}")
    
    # Tab 2: Association Rules
    with tab2:
        st.header("🔍 Association Rules Discovery")
        
        with st.spinner("Mining frequent itemsets and generating association rules..."):
            # Run analysis
            frequent_itemsets, rules = run_analysis(analyzer, min_support, algorithm, metric, min_threshold)
            
            if len(frequent_itemsets) == 0:
                st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support threshold.")
            else:
                st.success(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
                
                if len(rules) == 0:
                    st.warning("⚠️ No association rules found. Try lowering the minimum threshold.")
                else:
                    st.success(f"✅ Generated {len(rules)} association rules")
                    
                    # Display top rules
                    st.subheader("📊 Top Association Rules")
                    
                    sort_metric = st.selectbox("Sort by", ["lift", "confidence", "support", "conviction"])
                    top_rules_n = st.slider("Number of rules to display", 5, 50, 20, key="rules_display")
                    
                    display_rules = rules.nlargest(top_rules_n, sort_metric)
                    
                    # Format for display
                    display_df = display_rules[[
                        'antecedents_str', 'consequents_str', 
                        'support', 'confidence', 'lift', 'conviction'
                    ]].copy()
                    
                    display_df.columns = [
                        'Antecedents (IF)', 'Consequents (THEN)', 
                        'Support', 'Confidence', 'Lift', 'Conviction'
                    ]
                    
                    display_df = display_df.round(4)
                    st.dataframe(display_df, use_container_width=True, height=600)
                    
                    # Metrics explanation
                    with st.expander("ℹ️ Understanding the Metrics"):
                        st.markdown("""
                        - **Support**: How frequently the itemset appears in the dataset
                        - **Confidence**: How often the rule is correct (P(consequent|antecedent))
                        - **Lift**: How much more likely the consequent is given the antecedent
                          - Lift > 1: Positive correlation
                          - Lift = 1: No correlation
                          - Lift < 1: Negative correlation
                        - **Conviction**: Measures the implication strength (higher is stronger)
                        """)
                    
                    # Scatter plot of rules
                    st.markdown("---")
                    st.subheader("📊 Rules Visualization")
                    fig_scatter = visualizer.plot_rules_scatter(rules, 'support', 'confidence', 'lift')
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Download rules
                    csv = display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Rules as CSV",
                        data=csv,
                        file_name="association_rules.csv",
                        mime="text/csv"
                    )
    
    # Tab 3: Network Analysis
    with tab3:
        st.header("🕸️ Product Association Network")
        
        with st.spinner("Building association network..."):
            frequent_itemsets, rules = run_analysis(analyzer, min_support, algorithm, metric, min_threshold)
            
            if len(frequent_itemsets) > 0 and len(rules) > 0:
                network_top_n = st.slider("Number of top rules to visualize", 10, 100, 50, key="network_n")
                
                network_graph = analyzer.build_association_network(network_top_n)
                fig_network = visualizer.plot_association_network(network_graph, network_top_n)
                
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)
                    
                    st.info("""
                    **How to read this network:**
                    - Each node represents a product
                    - Arrows show associations (A → B means "if A, then B")
                    - Thicker lines indicate stronger associations (higher lift)
                    - Larger nodes have more connections
                    """)
                else:
                    st.warning("Unable to generate network visualization.")
            else:
                st.warning("⚠️ No association rules found for network visualization.")
    
    # Tab 4: Product Insights
    with tab4:
        st.header("🔥 Product Co-occurrence & Insights")
        
        # Co-occurrence heatmap
        st.subheader("Product Co-occurrence Heatmap")
        heatmap_top_n = st.slider("Number of products in heatmap", 10, 50, 30, key="heatmap_n")
        
        with st.spinner("Generating co-occurrence matrix..."):
            cooccurrence, product_names = analyzer.calculate_cooccurrence_matrix(heatmap_top_n)
            fig_cooccur = visualizer.plot_cooccurrence_heatmap(cooccurrence, product_names)
            st.plotly_chart(fig_cooccur, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - Darker colors indicate products that are frequently bought together
        - Diagonal shows how often each product appears
        - Use this to identify product bundling opportunities
        """)
        
        # Product statistics
        st.markdown("---")
        st.subheader("📊 Detailed Product Statistics")
        
        stats_df = analyzer.get_product_statistics()
        st.dataframe(stats_df, use_container_width=True, height=500)
        
        # Download stats
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Product Statistics",
            data=csv_stats,
            file_name="product_statistics.csv",
            mime="text/csv"
        )
    
    # Tab 5: Time Trends
    with tab5:
        st.header("⏰ Time-based Purchase Trends")
        
        with st.spinner("Analyzing temporal patterns..."):
            time_data = analyzer.get_time_based_data()
            
            fig_daily = visualizer.plot_daily_trends(time_data['daily'])
            st.plotly_chart(fig_daily, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hourly = visualizer.plot_hourly_distribution(time_data['hourly'])
                st.plotly_chart(fig_hourly, use_container_width=True)
            with col2:
                fig_dow = visualizer.plot_day_of_week(time_data['day_of_week'])
                st.plotly_chart(fig_dow, use_container_width=True)
        
        st.markdown("---")
        
        # Peak times analysis
        st.subheader("🕐 Peak Shopping Times")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Peak Hour", f"{time_data['peak_hour']}:00")
        with col2:
            st.metric("📈 Transactions at Peak", f"{time_data['peak_count']}")
        with col3:
            st.metric("📊 Avg Hourly Transactions", f"{time_data['avg_hourly']:.1f}")
        
        st.info("""
        **Business Insights:**
        - Identify peak shopping hours for staff scheduling
        - Plan promotions during high-traffic periods
        - Analyze day-of-week patterns for inventory management
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p>📊 Market Basket Analysis Dashboard | Built with Streamlit & MLxtend</p>
    <p>Use the sidebar to adjust analysis parameters and explore different insights</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
