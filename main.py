from src.data_loader import load_and_clean_data
from src.eda import visualize_target_distribution, plot_distributions, plot_correlation_heatmap
from src.trainer import train_model
from src.predictor import predict_user

# Load Data
df = load_and_clean_data("data/bot_detection_data.csv")

# Perform EDA
visualize_target_distribution(df)
plot_distributions(df)
plot_correlation_heatmap(df)

# Train Model
model, train_acc, test_acc = train_model(df)
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Predict on new input
sample_input = (254875, 55, 7, 5875, 1)
result = predict_user(model, sample_input)
print("Prediction for input:", result)
