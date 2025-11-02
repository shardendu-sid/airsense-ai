import io
import base64
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

def generate_base64_plot(data, variable, color, title, xlabel):
    plt.figure(figsize=(6, 4))
    sns.histplot(data[variable], kde=True, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    # Saving plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    # Encoding plot buffer to Base64
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_data

def generate_correlation_heatmap(data):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)

    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_data
