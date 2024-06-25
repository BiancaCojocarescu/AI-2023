import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(file_path, split_ratio=0.8):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) == 8:
                data.append(list(map(float, columns)))

    data = np.array(data)
    data = normalize_data(data)
    np.random.shuffle(data)
    split_idx = int(split_ratio * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]
    return train_data, test_data

def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_true, y_pred):
    error = np.mean((y_true - y_pred) ** 2)
    return error

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward_propagation(inputs, target_output, hidden_layer_output, output_layer_output, weights_hidden_output, weights_input_hidden, learning_rate):
    output_error = target_output - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

def train_neural_network(file_path, input_size, hidden_size, output_size, learning_rate, max_epochs):
    train_data, test_data = load_data(file_path)
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(max_epochs):
        total_error = 0
        for sample in train_data:
            if np.isnan(sample).any():
                print("NaN values detected in the current sample.")
                continue

            inputs = np.array([sample[:-1]])
            target_output = np.zeros((1, output_size))
            if np.isnan(sample[-1]):
                print("NaN values detected in the current sample.")
                continue
            else:
                target_output[0, int(sample[-1]) - 1] = 1

            hidden_output, predicted_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
            error = mean_squared_error(target_output, predicted_output)
            total_error += error

            weights_input_hidden, weights_hidden_output = backward_propagation(inputs, target_output, hidden_output, predicted_output, weights_hidden_output, weights_input_hidden, learning_rate)

        average_error = total_error / len(train_data)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Average Error: {average_error}')

    evaluate_performance(test_data, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

def predict(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    _, predicted_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    return predicted_output

def evaluate_performance(test_data, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    correct_predictions = 0
    total_samples = len(test_data)
    predicted_labels = []
    true_labels = []

    for sample in test_data:
        if np.isnan(sample).any():
            print("NaN values detected in the current sample.")
            continue

        inputs = np.array([sample[:-1]])
        target_output = np.zeros((1, output_size))
        if np.isnan(sample[-1]):
            print("NaN values detected in the current sample.")
            continue
        else:
            target_output[0, int(sample[-1]) - 1] = 1

        _, predicted_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        predicted_class = np.argmax(predicted_output)
        true_class = np.argmax(target_output)

        predicted_labels.append(predicted_class)
        true_labels.append(true_class)

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples

    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    print(f'Accuracy on test data: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Confusion Matrix:\n{confusion_mat}')

file_path = 'seeds_dataset.txt'
input_size = 7
hidden_size = 10
output_size = 3
learning_rate = 0.03
max_epochs = 1000

train_neural_network(file_path, input_size, hidden_size, output_size, learning_rate, max_epochs)