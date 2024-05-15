import plotly.graph_objs as go
import plotly.io as pio
import os

# Define a function to read data from a list of files
def read_data_from_files(file_list):
    data = []
    for file_path in file_list:
        with open(file_path, "r") as f:
            for line in f:
                result_dict = eval(line.strip())
                data.append(result_dict)
    return data

# List all files in the results directory
results_directory = "results"
file_list = [os.path.join(results_directory, file_name) for file_name in os.listdir(results_directory)]

# Read data from files
data = read_data_from_files(file_list)

# Group data by mode
mode_data = {}
for entry in data:
    mode = entry['mode']
    if mode not in mode_data:
        mode_data[mode] = {'state_space_sizes': [], 'solve_times': []}
    mode_data[mode]['state_space_sizes'].append(entry['map_size'][0] * entry['map_size'][1])
    mode_data[mode]['solve_times'].append(entry['Time Elapsed'])

# Plotting with Plotly
fig = go.Figure()

for mode, mode_info in mode_data.items():
    fig.add_trace(go.Scatter(
        x=mode_info['state_space_sizes'],
        y=mode_info['solve_times'],
        mode='markers',
        name=mode
    ))

fig.update_layout(
    title='State Space Size vs Solve Time for Different Modes',
    xaxis=dict(title='State Space Size'),
    yaxis=dict(title='Solve Time (seconds)'),
    showlegend=True,
    legend=dict(title='Modes'),
    hovermode='closest'
)

# Show the plot
pio.show(fig)
