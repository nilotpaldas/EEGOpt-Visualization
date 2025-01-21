import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import os

# Load and preprocess datasets
data1 = pd.read_csv('EEGOpt_Trials_ICM_Dataset.csv')
data2 = pd.read_csv('EEGOpt_Trials_NMEDE_Dataset.csv')
data3 = pd.read_csv('EEGOpt_Trials_NMEDM_Dataset.csv')

for data in [data1, data2, data3]:
    data['WPD_Threshold'] = data['WPD_Threshold'].round(2)
    data['LR_C'] = data['LR_C'].round(2)
    data['Denoiser'] = pd.Categorical(data['Denoiser'], categories=['EMD', 'WPD', 'NONE'], ordered=True)
    # Convert durations column to seconds
    data['Total_Time'] = pd.to_timedelta(data['Duration']).dt.total_seconds()

# Define plot titles, x-labels, and y-labels
plot_info = [
    ('Denoiser', 'Denoiser', 'Denoiser'),
    ('Feature Set', 'Feature', 'Feature Set'),
    ('PCA', 'PCA_param', 'PCA Explained Variance'),
    ('Classifier', 'Classifier', 'Classifier'),
    ('WPD Threshold', 'WPD_Threshold', 'WPD Threshold'),
    ('WPD Wavelet', 'WPD_Wavelet', 'WPD Wavelet'),
    ('EMD Threshold', 'EMD_Threshold', 'EMD Threshold'),
    ('Entropy Delay', 'Entropy_Delay', 'Delay'),
    ('Entropy Order', 'Entropy_Delay', 'Order'),
    ('HFD Kmax', 'HFD_Kmax', 'HFD Kmax'),
    ('SVM Kernel', 'SVM_Kernel', 'SVM Kernel'),
    ('KNN Neighbors', 'KNN_Neighbors', 'KNN Neighbors'),
    ('MLP Hidden Layer Units', 'MLP_Units', 'MLP Hidden Layer Units'),
    ('RF Num Trees', 'RF_Estimators', 'RF Number of Trees'),
    ('LR Regularization', 'LR_C', 'LR Regularization')
]

# Function to generate heatmaps
def generate_heatmaps(data):
    heatmaps = []
    heatmap_info = [
        ('Classifier vs. Denoiser', 'Classifier', 'Denoiser'),
        ('Classifier vs. Feature Set', 'Classifier', 'Feature'),
        ('Denoiser vs. Feature Set', 'Denoiser', 'Feature')
    ]

    for title, x_param, y_param in heatmap_info:
        pivot_table = data.pivot_table(
            index=y_param, columns=x_param, values='Objective_Value', aggfunc=['mean', 'median', 'max', 'min']
        )
        mean_values = pivot_table['mean']
        median_values = pivot_table['median']
        max_values = pivot_table['max']
        min_values = pivot_table['min']

        fig = go.Figure(
            data=go.Heatmap(
                z=max_values.values,
                x=max_values.columns,
                y=max_values.index,
                text=[
                    [
                        f"{max_values.loc[row, col]:.2f}"
                        for col in max_values.columns
                    ]
                    for row in max_values.index
                ],
                hovertext=[
                    [
                        f"Mean: {mean_values.loc[row, col]:.2f}<br>"
                        f"Median: {median_values.loc[row, col]:.2f}<br>"
                        f"Max: {max_values.loc[row, col]:.2f}<br>"
                        f"Min: {min_values.loc[row, col]:.2f}"
                        for col in max_values.columns
                    ]
                    for row in max_values.index
                ],
                texttemplate="%{text}",
                colorscale='viridis',
                colorbar=dict(title="Objective Value")
            )
        )
        fig.update_layout(
            height=450,
            width=600,
            title_text=title
        )
        heatmaps.append(dcc.Graph(figure=fig))
    return heatmaps

# Function to generate scatter plots
def generate_plots(data):
    plots = []
    for title, param, xlabel in plot_info:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[param],
                y=data['Objective_Value'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=data['Objective_Value'],
                    colorscale='viridis',
                    cmin=data['Objective_Value'].min(),
                    cmax=data['Objective_Value'].max(),
                    showscale=True,
                    colorbar=dict(title="Objective Value"),
                ),
                hovertemplate=(
                    f"{xlabel}: %{{x}}<br>"
                    f"Objective Value: %{{y}}<br>"
                    f"Denoiser: %{{customdata[0]}}<br>"
                    f"Feature Set: %{{customdata[1]}}<br>"
                    f"Classifier: %{{customdata[2]}}<extra></extra>"
                ),
                customdata=data[['Denoiser', 'Feature', 'Classifier']].to_numpy()
            )
        )

        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text="Objective Value")
        fig.update_layout(
            height=450,
            width=600,
            title_text=title,
            showlegend=False
        )
        plots.append(dcc.Graph(figure=fig))
    return plots

# Function to generate additional plots
def generate_additional_plots(data, sampler, random_seed):
    data = data[(data['Sampler'] == sampler) & (data['Random_Seed'] == random_seed)].copy()
    # Subtract the minimum cache size from all values in the Cache_Size column
    data['Cache_Size'] -= data['Cache_Size'].min()

    data['Cache_Size'] = data['Cache_Size'].interpolate()

    additional_plots = []

    for column, title, color, y_label in [
        ('Cache_Size', 'Cache Generated vs. Trial Number', 'blue', 'Cache Generated (GB)'),
        ('SignalProcessing_Time', 'Signal Processing Time vs. Trial Number', 'green', 'Time (seconds)'),
        ('Classification_Time', 'Classification Time vs. Trial Number', 'red', 'Time (seconds)'),
        ('Total_Time', 'Total Time vs. Trial Number', 'purple', 'Total Time (seconds)')
    ]:
        # Convert Cache_Size to GB if applicable
        if column == 'Cache_Size':
            data[column] = data[column] / 1024

        fig = go.Figure()

        if column == 'Cache_Size':
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='lines',
                    line=dict(color=color),
                    name=f"{title.split(' vs.')[0]} (Line)"
                )
            )
        else:
            # Add bar plot
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[column],
                    marker_color=color,
                    name=f"{title.split(' vs.')[0]} (Bar)"
                )
            )

            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=f"{title.split(' vs.')[0]} (Scatter)",
                    hovertemplate=(
                        f"Trial Number: %{{x}}<br>"
                        f"{title.split(' vs.')[0]}: %{{y}} {y_label}<br>"
                        f"Classifier: %{{customdata[0]}}<br>"
                        f"Feature Set: %{{customdata[1]}}<br>"
                        f"Denoiser: %{{customdata[2]}}<br>"
                        f"Sampler: %{{customdata[3]}}<extra></extra>"
                    ),
                    customdata=data[['Classifier', 'Feature', 'Denoiser', 'Sampler']].to_numpy()
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Trial Number",
            yaxis_title=y_label,
            height=450,
            width=600,
            barmode='overlay',
            showlegend=False
        )
        additional_plots.append(dcc.Graph(figure=fig))

    return additional_plots

# Create a Dash app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("EEGOpt Search Visualization", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Select Dataset:", style={'fontSize': '18px', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': 'Dataset 1: ICM', 'value': 'dataset1'},
                {'label': 'Dataset 2: NMED-E', 'value': 'dataset2'},
                {'label': 'Dataset 3: NMED-M', 'value': 'dataset3'}
            ],
            value='dataset1',
            style={'width': '300px', 'marginBottom': '20px', 'marginLeft': 'auto', 'marginRight': 'auto'}
        )
    ], style={'textAlign': 'center'}),
    html.H2(id='dataset-title', style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div("The following heatmaps display the maximum objective values for different parameter combinations across "
             "all the evaluated samplers:",
             style={'textAlign': 'center', 'marginBottom': '5px'}),
    html.Div("Objective Values refer to average cross-validation MCC Scores for each trial.",
             style={'textAlign': 'center', 'marginBottom': '10px'}),
    html.Div(id='heatmaps-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    html.Div("Below are scatter plots illustrating the relationship between various parameters and objective values:",
             style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}),
    html.Div(id='plots-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    html.H3("Cache and Timing Analysis by Sampler", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div("Select a sampler and random state to view Cache Generated, Signal Processing Time, Classification Time, and Total Time trends:",
             style={'textAlign': 'center', 'marginBottom': '10px'}),
    html.Div([
        dcc.Dropdown(
            id='sampler-dropdown',
            options=[],
            style={'width': '300px', 'marginBottom': '20px', 'marginLeft': 'auto', 'marginRight': 'auto'}
        ),
        dcc.Dropdown(
            id='random-state-dropdown',
            options=[],
            style={'width': '300px', 'marginBottom': '20px', 'marginLeft': 'auto', 'marginRight': 'auto'}
        )
    ], style={'textAlign': 'center'}),
    html.Div(id='timing-plots-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    html.Div("Finally, a table is provided for a detailed view of the search processes:",
             style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '10px'}),
    html.Div(id='table-container', style={'marginTop': '20px', 'textAlign': 'center'})
])

# Define the callback for dataset selection
@app.callback(
    [Output('dataset-title', 'children'),
     Output('heatmaps-container', 'children'),
     Output('plots-container', 'children'),
     Output('timing-plots-container', 'children'),
     Output('table-container', 'children'),
     Output('sampler-dropdown', 'options'),
     Output('sampler-dropdown', 'value'),
     Output('random-state-dropdown', 'options'),
     Output('random-state-dropdown', 'value')],
    [Input('dataset-dropdown', 'value'),
     Input('sampler-dropdown', 'value'),
     Input('random-state-dropdown', 'value')]
)
def update_visualization(selected_dataset, selected_sampler, selected_random_state):
    if selected_dataset == 'dataset1':
        data = data1
        title = "Dataset 1: ICM"
    elif selected_dataset == 'dataset2':
        data = data2
        title = "Dataset 2: NMED-E"
    elif selected_dataset == 'dataset3':
        data = data3
        title = "Dataset 3: NMED-M"
    else:
        data = data1
        title = "Dataset 1"

    heatmaps = generate_heatmaps(data)
    scatter_plots = generate_plots(data)

    sampler_options = [{'label': sampler, 'value': sampler} for sampler in data['Sampler'].unique()]
    selected_sampler = selected_sampler if selected_sampler in data['Sampler'].unique() else sampler_options[0]['value']

    random_state_options = [{'label': state, 'value': state} for state in data['Random_Seed'].unique()]
    selected_random_state = selected_random_state if selected_random_state in data['Random_Seed'].unique() else random_state_options[0]['value']

    additional_plots = generate_additional_plots(data, selected_sampler, selected_random_state)

    table = dash_table.DataTable(
        data=data.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in data.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'fontWeight': 'bold'},
    )

    return title, heatmaps, scatter_plots, additional_plots, table, sampler_options, selected_sampler, random_state_options, selected_random_state

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run_server(debug=True, host="0.0.0.0", port=port)

