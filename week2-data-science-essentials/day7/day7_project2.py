# Use another dataset and apply the same EDA steps in project 1
# Task 5: Explore advanced visualizations like boxplots or pair plots in Seaborn
# Task 6: Create a dashboard for your findings using Plotly or Dash


# sample example of plotly
import plotly.graph_objects as go

# Sample data
x = [1.5, 2.9, 3, 4.2, 5.6]
y = [2.2, 13.3, 4.4, 55.3, 52.1]

# Initialize a figure
fig = go.Figure()

# Add the scatter trace
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
))

# Show the plot
fig.show()