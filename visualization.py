"""
Visualization Module

This module contains all visualization functions for the market basket analysis dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np


class MarketBasketVisualizer:
    """Class containing all visualization methods"""
    
    @staticmethod
    def plot_top_products(products_df, top_n=20):
        """
        Plot horizontal bar chart of top products
        
        Args:
            products_df (pd.DataFrame): DataFrame with Product and Count columns
            top_n (int): Number of products to display
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        data = products_df.head(top_n)
        
        fig = px.bar(
            data, 
            x='Count', 
            y='Product', 
            orientation='h',
            title=f'Top {top_n} Most Frequently Purchased Products',
            color='Count',
            color_continuous_scale='Blues',
            labels={'Count': 'Purchase Frequency', 'Product': 'Product Name'}
        )
        
        fig.update_layout(
            height=600, 
            yaxis={'categoryorder': 'total ascending'},
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def plot_cooccurrence_heatmap(cooccurrence_matrix, product_names):
        """
        Create heatmap of product co-occurrence
        
        Args:
            cooccurrence_matrix (np.array): Co-occurrence matrix
            product_names (list): List of product names
            
        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=cooccurrence_matrix,
            x=product_names,
            y=product_names,
            colorscale='YlOrRd',
            colorbar=dict(title="Co-occurrence Count"),
            hoverongaps=False,
            hovertemplate='%{y} & %{x}<br>Co-occurrences: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Product Co-occurrence Heatmap (Top {len(product_names)} Products)',
            xaxis_title='Product',
            yaxis_title='Product',
            height=700,
            xaxis={'tickangle': -45},
            yaxis={'tickangle': 0}
        )
        
        return fig
    
    @staticmethod
    def plot_association_network(network_graph, top_n=50):
        """
        Create interactive network graph of product associations
        
        Args:
            network_graph (networkx.DiGraph): Network graph
            top_n (int): Number of rules used
            
        Returns:
            plotly.graph_objects.Figure: Network graph figure
        """
        if network_graph is None or len(network_graph.nodes()) == 0:
            return None
        
        # Create positions using spring layout
        pos = nx.spring_layout(network_graph, k=2, iterations=50, seed=42)
        
        # Create edge traces
        edge_traces = []
        for edge in network_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_weight = network_graph[edge[0]][edge[1]]['weight']
            confidence = network_graph[edge[0]][edge[1]]['confidence']
            
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge_weight * 0.5, color='#888'),
                    hovertemplate=f'{edge[0]} → {edge[1]}<br>Lift: {edge_weight:.2f}<br>Confidence: {confidence:.2f}<extra></extra>',
                    showlegend=False
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_hover = []
        
        for node in network_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Size based on degree (number of connections)
            degree = network_graph.degree(node)
            node_size.append(20 + degree * 5)
            node_hover.append(f'{node}<br>Connections: {degree}')
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=node_size,
                color='#1f77b4',
                line=dict(width=2, color='white')
            ),
            hovertext=node_hover,
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=f'Product Association Network (Top {top_n} Rules by Lift)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def plot_daily_trends(daily_data):
        """
        Plot daily transaction volume
        
        Args:
            daily_data (pd.DataFrame): Daily transaction counts
            
        Returns:
            plotly.graph_objects.Figure: Line chart
        """
        fig = px.line(
            daily_data, 
            x='date', 
            y='count',
            title='Daily Transaction Volume',
            labels={'date': 'Date', 'count': 'Number of Transactions'}
        )
        
        fig.update_traces(line_color='#1f77b4', line_width=2)
        fig.update_layout(
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def plot_hourly_distribution(hourly_data):
        """
        Plot hourly transaction distribution
        
        Args:
            hourly_data (pd.DataFrame): Hourly transaction counts
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        fig = px.bar(
            hourly_data,
            x='hour',
            y='count',
            title='Transactions by Hour of Day',
            labels={'hour': 'Hour', 'count': 'Number of Transactions'},
            color='count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        return fig
    
    @staticmethod
    def plot_day_of_week(dow_data):
        """
        Plot day of week transaction distribution
        
        Args:
            dow_data (pd.DataFrame): Day of week transaction counts
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        fig = px.bar(
            dow_data,
            x='day_of_week',
            y='count',
            title='Transactions by Day of Week',
            labels={'day_of_week': 'Day', 'count': 'Number of Transactions'},
            color='count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    @staticmethod
    def plot_rules_scatter(rules_df, x_metric='support', y_metric='confidence', color_metric='lift'):
        """
        Create scatter plot of association rules
        
        Args:
            rules_df (pd.DataFrame): Association rules dataframe
            x_metric (str): Metric for x-axis
            y_metric (str): Metric for y-axis
            color_metric (str): Metric for color
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot
        """
        if len(rules_df) == 0:
            return None
        
        fig = px.scatter(
            rules_df,
            x=x_metric,
            y=y_metric,
            color=color_metric,
            size='support',
            hover_data=['antecedents_str', 'consequents_str'],
            title=f'Association Rules: {y_metric.capitalize()} vs {x_metric.capitalize()}',
            color_continuous_scale='Viridis',
            labels={
                x_metric: x_metric.capitalize(),
                y_metric: y_metric.capitalize(),
                color_metric: color_metric.capitalize()
            }
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    @staticmethod
    def plot_itemset_distribution(frequent_itemsets):
        """
        Plot distribution of itemset lengths
        
        Args:
            frequent_itemsets (pd.DataFrame): Frequent itemsets dataframe
            
        Returns:
            plotly.graph_objects.Figure: Bar chart
        """
        if len(frequent_itemsets) == 0:
            return None
        
        length_counts = frequent_itemsets['length'].value_counts().sort_index()
        
        fig = px.bar(
            x=length_counts.index,
            y=length_counts.values,
            title='Distribution of Itemset Lengths',
            labels={'x': 'Number of Items in Itemset', 'y': 'Count'},
            color=length_counts.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400)
        
        return fig


class MetricsDisplay:
    """Class for creating metric display cards"""
    
    @staticmethod
    def format_metric_card(label, value, delta=None):
        """
        Format a metric for display
        
        Args:
            label (str): Metric label
            value (str/int/float): Metric value
            delta (str/int/float): Optional change value
            
        Returns:
            dict: Formatted metric data
        """
        return {
            'label': label,
            'value': value,
            'delta': delta
        }
    
    @staticmethod
    def get_rule_quality_badge(lift, confidence):
        """
        Get quality badge for a rule based on lift and confidence
        
        Args:
            lift (float): Lift value
            confidence (float): Confidence value
            
        Returns:
            str: Quality badge (Excellent, Good, Fair, Poor)
        """
        if lift > 2.0 and confidence > 0.7:
            return "🌟 Excellent"
        elif lift > 1.5 and confidence > 0.6:
            return "✅ Good"
        elif lift > 1.2 and confidence > 0.5:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
